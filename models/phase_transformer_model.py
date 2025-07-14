import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_model, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_model, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_model, bias=False)

        self.fc = nn.Linear(n_head * d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_model ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, self.n_head,  self.d_model)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head,  self.d_model)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head,  self.d_model)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, attn = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_embedding=64, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_model)
        self.w_2 = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model,  dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
class Encoder(nn.Module):

    def __init__(self,  n_layers, n_head, d_model, dropout=0.1, n_position=200):
        super().__init__()
        self.sensor_embedding = nn.Linear(d_model, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, dropout=dropout)
            for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.std = 0.02

    def forward(self, src_seq):

        emb_vector = self.sensor_embedding(src_seq)
        enc_output = self.dropout(self.position_enc(emb_vector))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output)

        return enc_output, _
    
class PhaseTransformerNeuralSignalModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        jitter_std: float = 0.0,
        nun_class: int = 4,
        phase_classes: list = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        dropout = 0.1
        self.phase_classes = phase_classes or []
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer_encoder = Encoder(
            n_position=200,
            d_model=d_model,
            n_layers=num_layers, 
            n_head=nhead,
            dropout=dropout)

        # Phase prediction head (binary classification) - only if phase_classes is not empty
        if len(self.phase_classes) > 0:
            self.phase_head = nn.Sequential(
                nn.Linear(d_model * 64 + d_model * 64,  d_model*2),
                nn.LeakyReLU(),
                nn.Linear(d_model*2,  d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 2),
                nn.Sigmoid()
            )  # 2 classes for phase
        else:
            self.phase_head = None
        
        # Class prediction head (nun_class classes)
        if len(self.phase_classes) > 0:
            self.class_head = nn.Sequential(
                nn.Linear(d_model * 64 + d_model * 64 + 2,  d_model*2),
                nn.Dropout(0.1),
                nn.LeakyReLU(),
                nn.Linear(d_model*2,  d_model),
                nn.Dropout(0.1),
                nn.LeakyReLU(),
                nn.Linear(d_model, nun_class),
                nn.Softmax(dim=-1)
            )
        else:
            # Classification only - no phase input
            self.class_head = nn.Sequential(
                nn.Linear(d_model * 64 + d_model * 64,  d_model*2),
                nn.Dropout(0.1),
                nn.LeakyReLU(),
                nn.Linear(d_model*2,  d_model),
                nn.Dropout(0.1),
                nn.LeakyReLU(),
                nn.Linear(d_model, nun_class),
                nn.Softmax(dim=-1)
            )
        
        # Loss functions
        self.phase_criterion = nn.BCELoss()
        self.class_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Learning rate and weight decay
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Data augmentation
        self.jitter_std = jitter_std
        self.input_dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape

        # Add jitter if in training mode
        x = self.input_dropout(x)
        
        # Project input to d_model dimensions
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        res_x = x.clone()

        # Apply transformer encoder
        x, _ = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Get phase and class predictions
        cat_hidden = torch.cat([x.reshape(x.shape[0],-1), res_x.reshape(res_x.shape[0] ,-1)], -1)
        
        if self.phase_head is not None:
            phase_logits = self.phase_head(cat_hidden)  # (batch_size, 2)
            cat_hidden = torch.cat([cat_hidden, phase_logits], -1)
        else:
            # Create dummy phase logits for compatibility
            phase_logits = torch.zeros(batch_size, 2, device=x.device)
        
        class_logits = self.class_head(cat_hidden)  # (batch_size, nun_class)
        
        return phase_logits, class_logits
    
    def training_step(self, batch, batch_idx):
        self.train()

        x, phase_target, class_target = batch
        
        # Move targets to the correct device
        phase_target = phase_target.float().to(self.device)
        class_target = class_target.to(self.device)
        
        # Apply jittering augmentation during training
        if self.training:
            noise = torch.randn_like(x) * 3
            x = x + noise
        
        # Get predictions
        phase_logits, class_logits = self(x)
        
        # Calculate losses
        if len(self.phase_classes) > 0:
            phase_loss = self.phase_criterion(phase_logits, phase_target)
        else:
            phase_loss = torch.tensor(0.0, device=self.device)
        
        # For class prediction, only consider samples where class_target != -1
        valid_class_mask = class_target != -1
        if valid_class_mask.any():
            valid_class_logits = class_logits[valid_class_mask]
            valid_class_target = class_target[valid_class_mask].long()  # Convert to long
            class_loss = nn.CrossEntropyLoss()(valid_class_logits, valid_class_target)
        else:
            class_loss = torch.tensor(0.0, device=self.device)
        
        # Combined loss
        total_loss = phase_loss + class_loss
        
        # Calculate accuracies
        if len(self.phase_classes) > 0:
            phase_preds = torch.sigmoid(phase_logits) > 0.5
            phase_acc = (phase_preds == phase_target).float().mean() * 100
            self.log('train_phase_acc', phase_acc, on_step=True, on_epoch=True, prog_bar=True)
        else:
            phase_acc = torch.tensor(0.0, device=self.device)
        
        if valid_class_mask.any():
            class_preds = torch.argmax(valid_class_logits, dim=1)
            class_acc = (class_preds == valid_class_target).float().mean() * 100
            self.log('train_class_acc', class_acc, on_step=True, on_epoch=True, prog_bar=True)
        else:
            class_acc = torch.tensor(0.0, device=self.device)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_phase_loss', phase_loss, on_step=True, on_epoch=True)
        self.log('train_class_loss', class_loss, on_step=True, on_epoch=True)
        # Log learning rate
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()

        x, phase_target, class_target = batch
        
        # Move targets to the correct device
        phase_target = phase_target.float().to(self.device)
        class_target = class_target.to(self.device)
        
        # Get predictions
        phase_logits, class_logits = self(x)
        
        # Calculate phase loss
        if len(self.phase_classes) > 0:
            phase_loss = self.phase_criterion(phase_logits, phase_target)
        else:
            phase_loss = torch.tensor(0.0, device=self.device)
        
        # For class prediction, only consider samples where class_target != -1
        valid_class_mask = class_target != -1
        if valid_class_mask.any():
            valid_class_logits = class_logits[valid_class_mask]
            valid_class_target = class_target[valid_class_mask].long()  # Convert to long
            class_loss = nn.CrossEntropyLoss()(valid_class_logits, valid_class_target)
        else:
            class_loss = torch.tensor(0.0, device=self.device)
        
        # Combined loss
        total_loss = phase_loss + class_loss
        
        # Calculate accuracies
        if len(self.phase_classes) > 0:
            phase_preds = torch.sigmoid(phase_logits) > 0.5
            phase_acc = (phase_preds == phase_target).float().mean() * 100
            self.log('val_phase_acc', phase_acc, on_step=True, on_epoch=True, prog_bar=True)
        else:
            phase_acc = torch.tensor(0.0, device=self.device)
        
        if valid_class_mask.any():
            class_preds = torch.argmax(valid_class_logits, dim=1)
            class_acc = (class_preds == valid_class_target).float().mean() * 100
            self.log('val_class_acc', class_acc, on_step=True, on_epoch=True, prog_bar=True)
        else:
            class_acc = torch.tensor(0.0, device=self.device)
        
        # Log metrics
        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_phase_loss', phase_loss, on_step=True, on_epoch=True)
        self.log('val_class_loss', class_loss, on_step=True, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=2e-4,
        )
        
        # Calculate total training steps
        num_training_steps = self.trainer.estimated_stepping_batches
        
        # Create cosine annealing scheduler with warm restarts
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=num_training_steps // 10,  # Restart every 10% of total steps
                T_mult=2,  # Double the restart interval after each restart
                eta_min=5e-5  # Minimum learning rate
            ),
            "interval": "step",
            "frequency": 1
        }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }


import math    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, n_position=200):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(n_position, d_model)
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].clone().detach()