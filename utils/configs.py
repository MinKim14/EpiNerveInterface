from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseConfig:
    """Base configuration for all models"""
    window_size: int = 64
    stride: int = 1
    batch_size: int = 256
    num_workers: int = 0
    max_epochs: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    jitter_std: float = 3.0
    seed: int = 42

@dataclass
class LSTMConfig(BaseConfig):
    """Configuration for LSTM model"""
    hidden_size: int = 128
    num_layers: int = 5
    dropout: float = 0.2
    bidirectional: bool = True

@dataclass
class TransformerConfig(BaseConfig):
    """Configuration for Transformer model"""
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 5
    dim_feedforward: int = 512
    dropout: float = 0.2
    batch_size: int = 256

def get_lstm_config(data_type: str = "raw", split_type: str = "sequential") -> LSTMConfig:
    """Get LSTM configuration based on data type and split type"""
    if data_type == "raw":
        if split_type == "sequential":
            return LSTMConfig(
                window_size=64,
                hidden_size=128,
                num_layers=3,
                dropout=0.1,
                bidirectional=True
            )
        else:  # subject split
            return LSTMConfig(
                window_size=64,
                hidden_size=128,
                num_layers=3,
                dropout=0.1,
                bidirectional=True
            )
    else:  # processed data
        return LSTMConfig(
            window_size=64,
            hidden_size=128,
            num_layers=3,
            dropout=0.6,
            bidirectional=True
        )

def get_transformer_config(data_type: str = "raw", split_type: str = "sequential") -> TransformerConfig:
    """Get Transformer configuration based on data type and split type"""
    if data_type == "raw":
        return TransformerConfig(
            window_size=64,
            d_model=128,
            nhead=8,
            num_layers=5,
            dim_feedforward=128,
            dropout=0.1,
            learning_rate=1e-4,
        )
    else:  # processed data
        return TransformerConfig(
            window_size=32,
            d_model=128,
            nhead=8,
            num_layers=5,
            dim_feedforward=128,
            dropout=0.1,
            learning_rate=1e-4,
        ) 