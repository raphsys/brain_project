import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
from core.memory_supervisor_v2 import MemorySupervisorV2

class MemoryInterfaceFAISS_Multi_V2:
    def __init__(self, dimension, num_classes=10):
        self.dimension = dimension
        self.num_classes = num_classes

        self.memory = {i: [] for i in range(num_classes)}
        self.index_dict = {i: faiss.IndexFlatL2(dimension) for i in range(num_classes)}
        self.supervisor = MemorySupervisorV2(num_classes)

    def store_trace(self, activations, labels):
        activations = activations.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        for vec, label in zip(activations, labels):
            self.memory[label].append(vec)

    def consolidate(self):
        for label in range(self.num_classes):
            vectors = self.memory[label]
            n_samples = len(vectors)
            if n_samples < 30:
                continue

            array = np.stack(vectors).astype(np.float32)
            var = np.var(array, axis=0)
            avg_var = np.mean(var)
            compression = 0.5
            target_size = max(5, int(n_samples * compression * (1 - avg_var)))

            kmeans = MiniBatchKMeans(n_clusters=target_size, batch_size=target_size*2, n_init=3)
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

    def query_similarities(self, activations, labels, k=5):
        activations = activations.detach().cpu().numpy().astype(np.float32)
        total_sim = []

        for vec, label in zip(activations, labels.cpu().numpy()):
            if not self.memory[label]:
                total_sim.append(0.0)
                continue
            vec = vec.reshape(1, -1)
            array = np.stack(self.memory[label]).astype(np.float32)
            index = faiss.IndexFlatL2(self.dimension)
            index.add(array)
            D, I = index.search(vec, min(k, len(array)))
            sim = 1.0 - np.mean(D)
            total_sim.append(sim)

        return np.mean(total_sim)

    def replay_samples(self):
        vectors, labels = [], []
        for label in range(self.num_classes):
            mem = self.memory[label]
            replay_count = self.supervisor.get_replay_budget(label)
            selected = mem[:min(replay_count, len(mem))]
            vectors += selected
            labels += [label] * len(selected)

        return np.stack(vectors), np.array(labels)

