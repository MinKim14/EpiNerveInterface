import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional

class BaseNeuralSignalModel(pl.LightningModule):
    """Base class for all neural signal models"""
    
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        jitter_std: float = 5,  # Standard deviation for jittering
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

    def on_validation_epoch_end(self):
        self.eval()
        
    def forward(self, x):
        """Forward pass of the model"""
        raise NotImplementedError
        
    def training_step(self, batch, batch_idx):
        """Training step with jittering augmentation"""
        x, y = batch
        # Apply jittering augmentation during training
        if self.training:
            # Add Gaussian noise to the input
            noise = torch.randn_like(x) * self.hparams.jitter_std
            x = x + noise
        
        # Forward pass
        logits = self(x)
        
        # Calculate loss
        loss = nn.CrossEntropyLoss()(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean() * 100
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        
        # Forward pass
        logits = self(x)
        # Calculate loss
        loss = nn.CrossEntropyLoss()(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean() * 100
        
        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        } 