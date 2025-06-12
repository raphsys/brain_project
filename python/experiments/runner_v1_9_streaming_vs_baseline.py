import os
import torch
from torchvision import datasets, transforms
from core.streaming_trainer_fast import StreamingTrainerFast
from core.adaptive_augmenter import get_pipeline
from core.balanced_sampler import create_balanced_subset
from core.flexible_model import FlexibleDeepMemoryNet
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dataset = datasets.MNIST('./data', train=True, download=True)

    ### === STREAMING BIOMEMORY === ###
    print("\n=== Streaming Biomemory ===")

    start_spc = 20
    max_spc = 50
    step_spc = 5
    stream_blocks = [start_spc] + list(range(start_spc + step_spc, max_spc + step_spc, step_spc))

    trainer = StreamingTrainerFast(model_arch=[128, 256], bottleneck_dim=64, device=device)

    for spc in stream_blocks:
        transform = get_pipeline(spc)
        augmented_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        subset = create_balanced_subset(augmented_dataset, spc)

        print(f"\n🚀 Injection streaming : {spc} samples par classe...")
        trainer.process_new_stream(subset, epochs=3)
        trainer.replay_and_refine()

    # Evaluation Biomemory
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=128
    )
    correct = 0
    total = 0
    trainer.model.eval()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, _ = trainer.model(images)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    accuracy_biomem = correct / total
    print(f"\n🎯 Final Biomemory Accuracy: {accuracy_biomem*100:.2f}%")

    ### === BASELINE CLASSIQUE === ###
    print("\n=== Baseline Classique ===")

    transform = get_pipeline(50)
    dataset_50 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    subset_50 = create_balanced_subset(dataset_50, 50)

    model_baseline = FlexibleDeepMemoryNet([128, 256], 64).to(device)
    optimizer = optim.Adam(model_baseline.parameters())
    loss_fn = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(subset_50, batch_size=128, shuffle=True)

    for epoch in range(10):
        model_baseline.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model_baseline(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model_baseline.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, _ = model_baseline(images)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    accuracy_baseline = correct / total
    print(f"\n🎯 Final Baseline Accuracy: {accuracy_baseline*100:.2f}%")

    ### === Résumé Comparatif === ###
    print("\n=== COMPARATIF FINAL ===")
    print(f"Biomemory: {accuracy_biomem*100:.2f}%")
    print(f"Baseline : {accuracy_baseline*100:.2f}%")
    print(f"Gain     : {(accuracy_biomem-accuracy_baseline)*100:+.2f}%")

