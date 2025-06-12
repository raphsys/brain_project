import os
import torch
from torchvision import datasets, transforms
from core.streaming_trainer import StreamingTrainer
from core.adaptive_augmenter import get_pipeline
from core.balanced_sampler import create_balanced_subset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db_path = os.path.join("db", "memory_streaming.db")
    os.makedirs("db", exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    # On prépare le dataset complet de base (sans transform appliqué encore)
    base_dataset = datasets.MNIST('./data', train=True, download=True)

    # Définir les blocs de stream successifs simulés
    stream_blocks = [3, 5, 10, 20, 50]

    trainer = StreamingTrainer(db_path, model_arch=[128, 256], bottleneck_dim=64, device=device)

    for spc in stream_blocks:
        transform = get_pipeline(spc)
        # On applique ici le transform avant le subset pour éviter le problème PIL
        augmented_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        subset = create_balanced_subset(augmented_dataset, spc)

        print(f"🚀 Injection streaming : {spc} samples par classe...")
        trainer.process_new_stream(subset, epochs=3)
        trainer.replay_and_refine()
        
                # Évaluation après chaque stream injection
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()), batch_size=64)
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

        

    print("✅ Streaming learning terminé.")

