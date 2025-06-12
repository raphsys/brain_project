import sqlite3
import numpy as np
import faiss

class MemoryInterfaceFAISS:
    def __init__(self, db_path, dimension):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.dimension = dimension

        self.buffer = []
        self.index = faiss.IndexFlatL2(dimension)
        self.labels = []

    def store_trace(self, activations, labels, epoch):
        activations = activations.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        for vec, label in zip(activations, labels):
            vec_bytes = vec.astype(np.float32).tobytes()
            self.cursor.execute("INSERT INTO traces(layer, vector, label, epoch, strength) VALUES (?, ?, ?, ?, ?)",
                                ("embedding", vec_bytes, int(label), epoch, 1.0))
            self.buffer.append((vec, label))

    def flush_buffer(self):
        self.conn.commit()

    def build_faiss_index(self):
        self.cursor.execute("SELECT vector, label FROM traces")
        data = self.cursor.fetchall()

        vectors = []
        labels = []
        for vec_bytes, label in data:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors.append(vec)
            labels.append(label)

        vectors = np.stack(vectors)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors.astype(np.float32))
        self.labels = np.array(labels)
        print(f"✅ FAISS Index reconstruit avec {len(vectors)} vecteurs.")

    def query_similarities(self, activations, k=5):
        activations = activations.detach().cpu().numpy().astype(np.float32)
        D, I = self.index.search(activations, k)
        mean_similarity = 1.0 - np.mean(D)
        return mean_similarity

    def consolidate(self):
        self.cursor.execute("SELECT vector, label FROM traces")
        data = self.cursor.fetchall()

        n_samples = len(data)

        if n_samples < 40:
            print(f"⚠️  Consolidation désactivée (only {n_samples} samples)")
            return

        vectors = []
        labels = []
        for vec_bytes, label in data:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors.append(vec)
            labels.append(label)

        vectors = np.stack(vectors).astype(np.float32)
        labels = np.array(labels)

        compression_ratio = min(0.5, 100 / n_samples)
        target_size = int(n_samples * compression_ratio)
        target_size = max(1, target_size)

        print(f"🚀 Consolidation mémoire sur {n_samples} → {target_size} vecteurs...")

        kmeans = faiss.Kmeans(self.dimension, target_size, niter=20, verbose=False)
        kmeans.train(vectors)

        centroids = kmeans.centroids

        _, assignments = faiss.knn(vectors, centroids, 1)
        new_labels = []
        for cluster_id in range(target_size):
            assigned_labels = labels[assignments.reshape(-1) == cluster_id]
            majority_label = np.bincount(assigned_labels).argmax()
            new_labels.append(majority_label)

        self.cursor.execute("DELETE FROM traces")
        for proto, label in zip(centroids, new_labels):
            vec_bytes = proto.astype(np.float32).tobytes()
            self.cursor.execute("INSERT INTO traces(layer, vector, label, epoch, strength) VALUES (?, ?, ?, ?, ?)",
                                ("embedding", vec_bytes, int(label), 0, 1.0))
        self.conn.commit()
        print(f"✅ Consolidation terminée. Nouvelle taille mémoire: {target_size}")

    # MODULE REPLAY HIPPOCAMPIQUE (v1.5)
    def replay_samples(self, num_samples):
        self.cursor.execute("SELECT vector, label FROM traces ORDER BY RANDOM() LIMIT ?", (num_samples,))
        data = self.cursor.fetchall()

        vectors = []
        labels = []
        for vec_bytes, label in data:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors.append(vec)
            labels.append(label)

        vectors = np.stack(vectors)
        labels = np.array(labels)
        return vectors, labels

