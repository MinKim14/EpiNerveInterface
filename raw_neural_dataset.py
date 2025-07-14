import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Optional, Literal, Dict
import pickle
from collections import Counter

class RawNeuralSignalDataset(Dataset):
    """Raw Neural Signal Dataset for PyTorch with sliding windows per file"""
    
    def __init__(
        self,
        data_dir: str,
        window_size: int = 32,
        stride: int = 1,
        split_type: str = "sequential",
        train_ratio: float = 0.8,
        seed: int = 42,
        cache_dir: str = "data/cache",
        train: bool = True,
        label_encoder: LabelEncoder = None
    ):
        """
        Args:
            data_dir (str): Directory containing the raw CSV files
            window_size (int): Size of the sliding window
            stride (int): Stride between windows
            split_type (str): Type of split ("sequential" or "subject")
            train_ratio (float): Ratio of training data (for sequential split)
            seed (int): Random seed for reproducibility
            cache_dir (str): Directory to store parsed data and window indices
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.split_type = split_type
        self.train_ratio = train_ratio
        self.seed = seed
        self.cache_dir = cache_dir
        self.train = train
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load or create label encoder
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            # First pass to fit label encoder
            self._load_data(fit_encoder=True)
            print("\nLabel to Index Mapping:")
            for label, index in zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))):
                print(f"Label {label} -> Index {index}")
        else:
            self.label_encoder = label_encoder
        
        # Second pass to load data with fitted encoder
        self._load_data(fit_encoder=False)
        
    def _get_window_indices(self, data_length: int, labels: np.ndarray) -> List[Tuple[int, int]]:
        """Generate window start and end indices"""
        indices = []
        window_labels = []
        for i in range(0, data_length - self.window_size + 1, self.stride):
            indices.append((i, i + self.window_size))
            window_labels.append(labels[i + self.window_size - 1])
        return indices, np.array(window_labels)
        
    def _parse_and_cache_file(self, file_path: str) -> Dict:
        """Parse a single file and cache its data and window indices"""
        # Generate cache filename
        cache_file = os.path.join(self.cache_dir, os.path.basename(file_path).replace('.csv', '.pkl'))
        
        # Read and parse the file
        df = pd.read_csv(file_path)
        
        # Separate features and labels
        features = df.iloc[:, :-1]  # All columns except the last one
        labels = df.iloc[:, -1]     # Last column as labels
        
        # Convert features to numeric values and handle any non-numeric data
        numeric_features = features.apply(pd.to_numeric, errors='coerce')
        numeric_features = numeric_features.fillna(0)
        
        # Convert labels to numeric using the pre-fitted label encoder
        labels = self.label_encoder.transform(labels)
        
        data_length = len(numeric_features)
        
        # Generate window indices
        window_indices, window_labels = self._get_window_indices(data_length, labels)
        
        # Create cache data
        cache_data = {
            'data_length': data_length,
            'num_channels': len(numeric_features.columns),
            'window_indices': window_indices,
            'file_path': file_path,
            'features': numeric_features.values.astype(np.float32),
            'labels': labels.astype(np.int64),
            'window_labels': window_labels.astype(np.int64)
        }
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        return cache_data
        
    def _load_data(self, fit_encoder: bool = False):
        """Load and preprocess all raw CSV files"""
        self.file_data = []  # List to store file metadata and window indices
        self.file_info = []  # Store (subject, week, trial) for each window
        
        # First pass: collect all unique labels
        all_labels = set()
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_raw_data.csv'):
                file_path = os.path.join(self.data_dir, filename)
                df = pd.read_csv(file_path)
                labels = df.iloc[:, -1]  # Last column as labels
                all_labels.update(labels.unique())
        
        # Initialize label encoder with all possible labels
        if fit_encoder:
            self.label_encoder.fit(list(all_labels))
        else:
            self.label_encoder.transform(list(all_labels))
        print(f"Found {len(all_labels)} unique classes: {list(all_labels)}")
        self.num_labels = len(all_labels)
        
        # Second pass: process files
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_raw_data.csv'):
                # Parse filename to get subject, week, and trial info
                parts = filename.split('_')
                subject = parts[0]
                week = parts[1].replace('w', '')
                trial = parts[2]
                
                file_path = os.path.join(self.data_dir, filename)
                print(f"Processing {filename}...")
                
                # Parse and cache file
                file_data = self._parse_and_cache_file(file_path)
                
                # Add file data and metadata
                self.file_data.append(file_data)
                self.file_info.extend([(subject, week, trial)] * len(file_data['window_indices']))
        
        # Create train/test split
        self._create_split()
        
    def _create_split(self):
        """Create train/test split based on split_type"""
        if self.split_type == "sequential":
            self._sequential_split()
        elif self.split_type == "subject":
            self._subject_split()
        else:
            raise ValueError(f"Unknown split type: {self.split_type}")

    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        # Find which file this window belongs to
        file_idx, st, ed = self.data_idx[idx]
        file_data = self.file_data[file_idx]
        window_features = file_data['features'][st:ed]
        label = self.data_label[idx]

        return window_features, label
    
    def _sequential_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split data sequentially for each class label"""
        # Get all labels and their indices
        current_idx = 0
        
        self.data_idx = []
        self.data_label = []

        for f_idx, file_data in enumerate(self.file_data):
            window_labels = file_data['window_labels']
            file_windows = file_data['window_indices']
            
            num_labels = [] 
            for label in range(self.num_labels):
                # Get indices for this label
                label_mask = window_labels == label
                label_indices = np.where(label_mask)[0]  # Get the actual indices where mask is True
                label_windows = [file_windows[i] for i in label_indices]  # Get corresponding windows
                num_labels.append(len(label_windows))
            print(num_labels)
            max_label_idx = np.argmax(num_labels)

            for i, label in enumerate(range(self.num_labels)):
                # Get indices for this label
                label_mask = window_labels == label
                label_indices = np.where(label_mask)[0]  # Get the actual indices where mask is True
                label_windows = [file_windows[i] for i in label_indices]  # Get corresponding windows
                split_idx = int(len(label_windows) * self.train_ratio)

                if(self.train):
                    cur_window = label_windows[:split_idx]
                    cur_window = [[f_idx, *cur_window[i]] for i in range(len(cur_window))]
                    cur_label = np.full(len(cur_window), label, dtype=np.int64)
                else:
                    cur_window = label_windows[split_idx:]
                    cur_window = [[f_idx, *cur_window[i]] for i in range(len(cur_window))]
                    cur_label = np.full(len(cur_window), label, dtype=np.int64)

                self.data_idx.extend(cur_window)
                self.data_label.extend(cur_label)

        self.data_idx = np.array(self.data_idx)
        self.data_label = np.array(self.data_label, dtype=np.int64)
    
    def _subject_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split data by subject (file)"""
        # Get unique files
        unique_files = set(data['file_path'] for data in self.file_data)
        
        # Calculate split point
        split_idx = int(len(unique_files) * self.train_ratio)
        
        # Split files
        train_files = list(unique_files)[:split_idx]
        test_files = list(unique_files)[split_idx:]

        self.data_idx = []
        self.data_label = []

        current_idx = 0
        for i, file_data in enumerate(self.file_data):
            window_labels = file_data['window_labels']
            file_windows = file_data['window_indices']

            if(self.train):
                if file_data['file_path'] in train_files:
                    cur_window = [[i, *window] for window in file_windows]  # Add file index to each window
                    cur_label = window_labels.astype(np.int64)
                else:
                    continue
            else:
                if file_data['file_path'] in test_files:
                    cur_window = [[i, *window] for window in file_windows]  # Add file index to each window
                    cur_label = window_labels.astype(np.int64)
                else:
                    continue
            
            self.data_idx.extend(cur_window)
            self.data_label.extend(cur_label)
            
        self.data_idx = np.array(self.data_idx)
        self.data_label = np.array(self.data_label, dtype=np.int64)

