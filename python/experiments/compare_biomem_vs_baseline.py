import os
import sys
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from core.flexible_model import FlexibleDeepMemoryNet
from core.memory_interface_faiss_multi import MemoryInterfaceFAISS_Multi as MemoryInterfaceFAISS
from core.balanced_sampler import create_balanced_subset
from core.adaptive_augmenter import get_pipeline

import numpy as np

def ask_int(prompt, default):
    user_input = input(f"{prompt} (default {default}): ")
    return int(user_input) if user_input.strip() else default

def ask_float(prompt, default):
    user_input = input(f"{prompt} (default {default}): ")
    return float(user_input) if user_input.strip() else default

def run_experiment(samples_per_class, epoch_count, hidden_layers, bottleneck_dim):

    transform = get_pipeline(samples_per_class)
    full_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    subset_dataset = create_balanced_subset(full_dataset, samples_per_class)
    train_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()

    ## Auto-adaptation mémoire
    lambda_memory = min(0.1, 1.0 / samples_per_class)
    early_stopping = False if samples_per_class < 20 else True
    threshold = 0.8
    replay_enabled = True

    ### BASELINE
    model_baseline = FlexibleDeepMemoryNet(hidden_layers, bottleneck_dim).to(device)
    optimizer = optim.Adam(model_baseline.parameters())
    for epoch in range(epoch_count):
        model_baseline.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model_baseline(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model_baseline.eval()
    correct = sum((model_baseline(images.to(device))[0].argmax(dim=1) == labels.to(device)).sum().item() for images, labels in test_loader)
    acc_baseline = correct / len(test_loader.dataset)

    ### BIOMEMORY MULTI-INDEX
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, '..', '..', 'db', 'memory.db')
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS traces (id INTEGER PRIMARY KEY, layer TEXT, vector BLOB, label INTEGER, epoch INTEGER, strength REAL)')
    conn.commit()
    conn.close()

    model_memory = FlexibleDeepMemoryNet(hidden_layers, bottleneck_dim).to(device)
    memory_interface = MemoryInterfaceFAISS(DB_PATH, dimension=bottleneck_dim, num_classes=10)
    optimizer = optim.Adam(model_memory.parameters())

    ## Pré-entraînement avant mémoire
    for _ in range(5):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model_memory(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, activ = model_memory(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        memory_interface.store_trace(activ, labels, epoch=0)

    memory_interface.flush_buffer()
    memory_interface.consolidate()
    memory_interface.build_faiss_index()

    for epoch in range(epoch_count):
        model_memory.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, activ = model_memory(images)

            with torch.no_grad():
                loss_mem = 1.0 - memory_interface.query_similarities(activ, labels, k=5)
                if early_stopping and loss_mem < (1 - threshold):
                    continue

            loss_classic = loss_fn(outputs, labels)
            probs = nn.functional.softmax(outputs, dim=1)
            max_confidence = probs.max(dim=1).values.mean().item()
            adaptive_lambda = lambda_memory * (1 - max_confidence)

            total_loss = loss_classic + adaptive_lambda * loss_mem
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if replay_enabled:
            replay_vectors, replay_labels = memory_interface.replay_samples()
            replay_vectors = torch.tensor(replay_vectors, dtype=torch.float32).to(device)
            replay_labels = torch.tensor(replay_labels, dtype=torch.long).to(device)
            logits_replay = model_memory.output_layer(replay_vectors)
            loss_replay = loss_fn(logits_replay, replay_labels)
            optimizer.zero_grad()
            loss_replay.backward()
            optimizer.step()

    model_memory.eval()
    correct_mem = sum((model_memory(images.to(device))[0].argmax(dim=1) == labels.to(device)).sum().item() for images, labels in test_loader)
    acc_memory = correct_mem / len(test_loader.dataset)

    return acc_baseline, acc_memory

# === Mode interactif ===

samples_per_class = ask_int("Nombre d'exemples par classe (max 6000)", 10)
epoch_count = ask_int("Nombre d'epochs d'entraînement", 5)
num_layers = ask_int("Combien de couches cachées", 2)
hidden_layers = [ask_int(f"Taille de la couche {i+1}", 128) for i in range(num_layers)]
bottleneck_dim = ask_int("Dimension du bottleneck mémoire", 64)

acc_base, acc_mem = run_experiment(samples_per_class, epoch_count, hidden_layers, bottleneck_dim)
gain = (acc_mem - acc_base)*100

print("=========================================")
print(f"Baseline accuracy : {acc_base:.4%}")
print(f"Biomemory accuracy: {acc_mem:.4%}")
print(f"Gain              : {gain:.2f}%")
print("=========================================")

