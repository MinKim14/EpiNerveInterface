import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from raw_neural_dataset import RawNeuralSignalDataset
from sklearn.preprocessing import LabelEncoder

class PhasePredictionDataset(Dataset):
    """Dataset for phase prediction with dual-task learning (phase + class)"""
    
    def __init__(
        self,
        data_dir: str,
        window_size: int = 100,
        stride: int = 10,
        split_type: str = "sequential",
        train_ratio: float = 0.8,
        seed: int = 42,
        cache_dir: str = "data/cache",
        train: bool = True,
        phase_classes: list = [],  # Classes that belong to phase 0 (e.g., immobile states)
        label_encoder: LabelEncoder = None
    ):
        # Initialize base dataset
        self.base_dataset = RawNeuralSignalDataset(
            data_dir=data_dir,
            window_size=window_size,
            stride=stride,
            split_type=split_type,
            train_ratio=train_ratio,
            seed=seed,
            cache_dir=cache_dir,
            train=train,
            label_encoder=label_encoder
        )
        
        self.phase_classes = phase_classes
        
    def _convert_labels(self, class_label):
        """Convert class labels to phase labels and reordered class labels"""
        # Phase label: [1,0] for phase 0 (classes in phase_classes), [0,1] for phase 1 (other classes)
        phase_label = [1, 0] if class_label in self.phase_classes else [0, 1]
        
        # Reorder class labels:
        # - Classes in phase_classes -> -1 (invalid for class prediction)
        # - Other classes -> 0,1,2,3 (in order of appearance)
        if class_label in self.phase_classes:
            class_label = -1  # Invalid for class prediction
        else:
            # Map remaining classes to 0,1,2,3
            remaining_classes = sorted([c for c in range(4) if c not in self.phase_classes])
            class_label = remaining_classes.index(class_label)
        
        return phase_label, class_label
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get features and original class label from base dataset
        features, class_label = self.base_dataset[idx]
        
        # Convert to phase and reordered class labels
        phase_label, class_label = self._convert_labels(class_label)
        
        return features, torch.tensor(phase_label, dtype=torch.float32), torch.tensor(class_label, dtype=torch.long)
    
    @property
    def label_encoder(self):
        return self.base_dataset.label_encoder
    
    @property
    def num_labels(self):
        return self.base_dataset.num_labels 