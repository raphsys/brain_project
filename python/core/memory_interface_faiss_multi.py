import sqlite3
import numpy as np
import faiss
import os
from core.memory_supervisor import MemorySupervisor

class MemoryInterfaceFAISS_Multi:
    def __init__(self, db_path, dimension, num_classes=10):
        self.db_path = db_path
        self.dimension = dimension
        self.num_classes = num_classes

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY,
            layer TEXT,
            vector BLOB,
            label INTEGER,
            epoch INTEGER,
            strength REAL)
        """)
        self.conn.commit()

        self.index_dict = {i: faiss.IndexFlatL2(dimension) for i in range(num_classes)}
        self.label_counts = {i: 0 for i in range(num_classes)}
        self.attention_masks = {i: np.ones(dimension, dtype=np.float32) for i in range(num_classes)}
        self.supervisor = MemorySupervisor(num_classes)

    def apply_attention(self, vec, label):
        return vec * self.attention_masks[label]

    def store_trace(self, activations, labels, epoch):
        activations = activations.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        for vec, label in zip(activations, labels):
            vec_att = self.apply_attention(vec, label)
            vec_bytes = vec_att.astype(np.float32).tobytes()
            self.cursor.execute("INSERT INTO traces(layer, vector, label, epoch, strength) VALUES (?, ?, ?, ?, ?)",
                                ("embedding", vec_bytes, int(label), epoch, 1.0))

    def flush_buffer(self):
        self.conn.commit()

    def build_faiss_index(self):
        self.cursor.execute("SELECT vector, label FROM traces")
        data = self.cursor.fetchall()

        vectors_by_class = {i: [] for i in range(self.num_classes)}
        for vec_bytes, label in data:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors_by_class[label].append(vec)

        for label, vectors in vectors_by_class.items():
            if vectors:
                array = np.stack(vectors).astype(np.float32)
                self.index_dict[label].reset()
                self.index_dict[label].add(array)
                self.label_counts[label] = len(array)

        total = sum(self.label_counts.values())
        print(f"✅ Multi-FAISS Index reconstruit avec {total} vecteurs répartis sur {self.num_classes} sous-mémoires.")

    def query_similarities(self, activations, labels, k=5):
        activations = activations.detach().cpu().numpy().astype(np.float32)
        total_sim = []

        for vec, label in zip(activations, labels.cpu().numpy()):
            if self.label_counts[label] == 0:
                total_sim.append(0.0)
                continue

            vec = self.apply_attention(vec, label).reshape(1, -1)
            D, I = self.index_dict[label].search(vec, min(k, self.label_counts[label]))
            sim = 1.0 - np.mean(D)
            total_sim.append(sim)

        return np.mean(total_sim)

    def consolidate(self):
        self.cursor.execute("SELECT vector, label FROM traces")
        data = self.cursor.fetchall()

        vectors_by_class = {i: [] for i in range(self.num_classes)}
        for vec_bytes, label in data:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors_by_class[label].append(vec)

        self.cursor.execute("DELETE FROM traces")

        for label in range(self.num_classes):
            vectors = vectors_by_class[label]
            n_samples = len(vectors)

            if n_samples < 40:
                print(f"⚠️  Consolidation désactivée pour classe {label} ({n_samples} samples)")
                for vec in vectors:
                    vec_bytes = vec.astype(np.float32).tobytes()
                    self.cursor.execute("INSERT INTO traces(layer, vector, label, epoch, strength) VALUES (?, ?, ?, ?, ?)",
                                         ("embedding", vec_bytes, int(label), 0, 1.0))
                continue

            array = np.stack(vectors).astype(np.float32)
            var = np.var(array, axis=0)
            beta = 2.0
            mask = np.exp(-beta * var)
            mask = mask / (np.max(mask) + 1e-8)
            self.attention_masks[label] = mask.astype(np.float32)

            avg_var = np.mean(var)
            compression_factor = 0.5
            target_size = max(5, int(n_samples * compression_factor * (1 - 0.8 * avg_var)))

            print(f"🚀 Consolidation classe {label}: {n_samples} → {target_size} vecteurs... (var={avg_var:.4f})")

            kmeans = faiss.Kmeans(self.dimension, target_size, niter=20, verbose=False)
            kmeans.train(array)
            centroids = kmeans.centroids

            for proto in centroids:
                vec_bytes = proto.astype(np.float32).tobytes()
                self.cursor.execute("INSERT INTO traces(layer, vector, label, epoch, strength) VALUES (?, ?, ?, ?, ?)",
                                     ("embedding", vec_bytes, int(label), 0, 1.0))

            self.supervisor.update_memory_info(label, avg_var, target_size)

    def light_consolidate(self):
        """
        Light consolidation for streaming.
        """
        self.cursor.execute("SELECT vector, label FROM traces")
        data = self.cursor.fetchall()

        vectors_by_class = {i: [] for i in range(self.num_classes)}
        for vec_bytes, label in data:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors_by_class[label].append(vec)

        self.cursor.execute("DELETE FROM traces")

        for label in range(self.num_classes):
            vectors = vectors_by_class[label]
            n_samples = len(vectors)
            if n_samples < 10:
                for vec in vectors:
                    vec_bytes = vec.astype(np.float32).tobytes()
                    self.cursor.execute("INSERT INTO traces(layer, vector, label, epoch, strength) VALUES (?, ?, ?, ?, ?)",
                                         ("embedding", vec_bytes, int(label), 0, 1.0))
                continue

            array = np.stack(vectors).astype(np.float32)
            light_target = max(5, int(n_samples * 0.7))
            kmeans = faiss.Kmeans(self.dimension, light_target, niter=10, verbose=False)
            kmeans.train(array)
            centroids = kmeans.centroids

            for proto in centroids:
                vec_bytes = proto.astype(np.float32).tobytes()
                self.cursor.execute("INSERT INTO traces(layer, vector, label, epoch, strength) VALUES (?, ?, ?, ?, ?)",
                                     ("embedding", vec_bytes, int(label), 0, 1.0))

            self.supervisor.update_memory_info(label, np.mean(np.var(array, axis=0)), light_target)

        self.conn.commit()

    def replay_samples(self):
        data = []
        for label in range(self.num_classes):
            self.cursor.execute("SELECT vector, label FROM traces WHERE label=? ORDER BY RANDOM() LIMIT ?", 
                                 (label, self.supervisor.get_replay_budget(label)))
            data += self.cursor.fetchall()

        vectors = []
        labels = []
        for vec_bytes, label in data:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors.append(vec)
            labels.append(label)

        vectors = np.stack(vectors)
        labels = np.array(labels)
        return vectors, labels

