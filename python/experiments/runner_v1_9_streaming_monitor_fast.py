import os
import torch
from torchvision import datasets, transforms
from core.streaming_trainer_fast import StreamingTrainerFast
from core.adaptive_augmenter import get_pipeline
from core.balanced_sampler import create_balanced_subset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dataset = datasets.MNIST('./data', train=True, download=True)
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

        # Evaluation
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
        accuracy = correct / total
        print(f"🎯 Accuracy après streaming SPC={spc}: {accuracy*100:.2f}%")

    print("\n✅ Streaming continual learning optimisé terminé.")

