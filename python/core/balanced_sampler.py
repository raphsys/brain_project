from collections import defaultdict
from torch.utils.data import Subset

def create_balanced_subset(dataset, samples_per_class):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    for cls in class_indices:
        if len(class_indices[cls]) < samples_per_class:
            raise ValueError(f"Pas assez d'exemples pour la classe {cls}")
    balanced_indices = []
    for cls, indices in class_indices.items():
        selected = indices[:samples_per_class]
        balanced_indices.extend(selected)
    return Subset(dataset, balanced_indices)

