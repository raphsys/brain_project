import os
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from core.flexible_model import FlexibleDeepMemoryNet
from core.balanced_sampler import create_balanced_subset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== EXPÉRIMENTATION BASELINE v2.0.3 ===")

    start_spc = int(input("SPC start: "))
    max_spc = int(input("SPC max: "))
    step_spc = int(input("Step SPC: "))

    num_layers = int(input("Nb couches cachées: "))
    hidden_layers = [int(input(f"Taille couche {i+1}: ")) for i in range(num_layers)]

    bottleneck_dim = int(input("Bottleneck dimension: "))

    stream_blocks = [start_spc] + list(range(start_spc + step_spc, max_spc + step_spc, step_spc))

    for spc in stream_blocks:
        print(f"\n🚀 Baseline training SPC={spc}...")

        # Préparation du dataset classique (sans attention, sans augmentation)
        transform = transforms.ToTensor()
        base_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        subset = create_balanced_subset(base_dataset, spc)

        loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True)

        model = FlexibleDeepMemoryNet(hidden_layers, bottleneck_dim).to(device)
        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(5):  # 5 epochs d'entraînement baseline simple
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                loss = loss_fn(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Évaluation sur test set complet
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True, transform=transform),
            batch_size=128
        )

        correct, total = 0, 0
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f"🎯 Baseline Accuracy SPC={spc}: {accuracy*100:.2f}%")

    print("\n✅ Baseline v2.0.3 terminé.")

