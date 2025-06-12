import torch.nn as nn

class FlexibleDeepMemoryNet(nn.Module):
    def __init__(self, hidden_layers, bottleneck_dim):
        super().__init__()

        self.flatten = nn.Flatten()

        self.hidden_layers = nn.ModuleList()
        in_features = 28*28
        for hidden_size in hidden_layers:
            self.hidden_layers.append(nn.Linear(in_features, hidden_size))
            self.hidden_layers.append(nn.ReLU())
            in_features = hidden_size

        self.embedding = nn.Linear(in_features, bottleneck_dim)
        self.embedding_act = nn.ReLU()
        self.output_layer = nn.Linear(bottleneck_dim, 10)

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.hidden_layers:
            x = layer(x)
        embedding = self.embedding_act(self.embedding(x))
        output = self.output_layer(embedding)
        return output, embedding

