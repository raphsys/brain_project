import torch
import torch.nn as nn
import torch.optim as optim
from core.flexible_model import FlexibleDeepMemoryNet
from core.memory_interface_faiss_multi_v2 import MemoryInterfaceFAISS_Multi_V2

class StreamingTrainerV2:
    def __init__(self, model_arch, bottleneck_dim, device):
        self.device = device
        self.model = FlexibleDeepMemoryNet(model_arch, bottleneck_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.CrossEntropyLoss()
        self.memory_interface = MemoryInterfaceFAISS_Multi_V2(dimension=bottleneck_dim)

    def process_new_stream(self, stream_dataset, epochs=3):
        loader = torch.utils.data.DataLoader(stream_dataset, batch_size=128, shuffle=True)
        for _ in range(epochs):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, activ = self.model(images)
                loss = self.loss_fn(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.memory_interface.store_trace(activ, labels)

        self.memory_interface.consolidate()
        self.memory_interface.build_faiss_index()

    def replay_and_refine(self):
        replay_vectors, replay_labels = self.memory_interface.replay_samples()
        replay_vectors = torch.tensor(replay_vectors, dtype=torch.float32).to(self.device)
        replay_labels = torch.tensor(replay_labels, dtype=torch.long).to(self.device)
        logits_replay = self.model.output_layer(replay_vectors)
        loss_replay = self.loss_fn(logits_replay, replay_labels)
        self.optimizer.zero_grad()
        loss_replay.backward()
        self.optimizer.step()

