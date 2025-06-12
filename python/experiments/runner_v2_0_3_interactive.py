import os
import torch
from torchvision import datasets, transforms
from core.streaming_trainer_v3 import StreamingTrainerV3
from core.adaptive_augmenter_v3 import get_pipeline
from core.balanced_sampler import create_balanced_subset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== EXPÉRIMENTATION v2.0.3 Optimisation Sensorielle ===")

    start_spc = int(input("SPC start: "))
    max_spc = int(input("SPC max: "))
    step_spc = int(input("Step SPC: "))

    num_layers = int(input("Nb couches cachées: "))
    hidden_layers = [int(input(f"Taille couche {i+1}: ")) for i in range(num_layers)]

    bottleneck_dim = int(input("Bottleneck dimension: "))

    stream_blocks = [start_spc] + list(range(start_spc + step_spc, max_spc + step_spc, step_spc))
    trainer = StreamingTrainerV3(model_arch=hidden_layers, bottleneck_dim=bottleneck_dim, device=device)

    # Prétraining sur tout MNIST avant streaming
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    trainer.pretrain_sensory(full_dataset)

    for spc in stream_blocks:
        transform = get_pipeline(spc)
        augmented_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        subset = create_balanced_subset(augmented_dataset, spc)

        print(f"\n🚀 Streaming : {spc} samples par classe...")
        trainer.process_new_stream(subset, epochs=3)
        trainer.replay_and_refine()

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
            batch_size=128
        )
        correct, total = 0, 0
        trainer.model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = trainer.model(images)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total
        print(f"🎯 Accuracy SPC={spc}: {accuracy*100:.2f}%")

    print("\n✅ v2.0.3 sensorielle terminé.")

