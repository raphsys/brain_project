import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
from core.memory_supervisor import MemorySupervisor

class MemoryInterfaceFAISS_Multi_Fast:
    def __init__(self, dimension, num_classes=10):
        self.dimension = dimension
        self.num_classes = num_classes

        self.memory = {i: [] for i in range(num_classes)}
        self.index_dict = {i: faiss.IndexFlatL2(dimension) for i in range(num_classes)}
        self.attention_masks = {i: np.ones(dimension, dtype=np.float32) for i in range(num_classes)}
        self.supervisor = MemorySupervisor(num_classes)

    def apply_attention(self, vec, label):
        return vec * self.attention_masks[label]

    def store_trace(self, activations, labels):
        activations = activations.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        for vec, label in zip(activations, labels):
            vec_att = self.apply_attention(vec, label)
            self.memory[label].append(vec_att)

    def consolidate(self):
        for label in range(self.num_classes):
            vectors = self.memory[label]
            n_samples = len(vectors)
            if n_samples < 40:
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

            kmeans = MiniBatchKMeans(n_clusters=target_size, batch_size=target_size*3, n_init=3)
            kmeans.fit(array)
            centroids = kmeans.cluster_centers_

            self.memory[label] = [proto for proto in centroids]
            self.supervisor.update_memory_info(label, avg_var, target_size)

    def build_faiss_index(self):
        for label in range(self.num_classes):
            vectors = self.memory[label]
            if vectors:
                array = np.stack(vectors).astype(np.float32)
                self.index_dict[label].reset()
                self.index_dict[label].add(array)
        total = sum(len(self.memory[label]) for label in range(self.num_classes))
        print(f"✅ Multi-FAISS Index reconstruit avec {total} vecteurs répartis sur {self.num_classes} sous-mémoires.")

    def query_similarities(self, activations, labels, k=5):
        activations = activations.detach().cpu().numpy().astype(np.float32)
        total_sim = []

        for vec, label in zip(activations, labels.cpu().numpy()):
            if not self.memory[label]:
                total_sim.append(0.0)
                continue

            vec = self.apply_attention(vec, label).reshape(1, -1)
            array = np.stack(self.memory[label]).astype(np.float32)
            index = faiss.IndexFlatL2(self.dimension)
            index.add(array)
            D, I = index.search(vec, min(k, len(array)))
            sim = 1.0 - np.mean(D)
            total_sim.append(sim)

        return np.mean(total_sim)

    def replay_samples(self):
        vectors = []
        labels = []
        for label in range(self.num_classes):
            mem = self.memory[label]
            replay_count = self.supervisor.get_replay_budget(label)
            selected = mem[:min(replay_count, len(mem))]
            vectors += selected
            labels += [label] * len(selected)

        vectors = np.stack(vectors)
        labels = np.array(labels)
        return vectors, labels

