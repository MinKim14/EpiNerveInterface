import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.base_model import BaseNeuralSignalModel

class PhaseLSTMNeuralSignalModel(BaseNeuralSignalModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        jitter_std: float = 5,
        num_class: int = 4,
        phase_classes: list = None,
    ):
        super().__init__(
            input_size=input_size,
            num_classes=num_class,  # For class prediction (0,1,2,3)
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            jitter_std=jitter_std
        )
        
        self.phase_classes = phase_classes or []
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate the size of LSTM output
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Phase prediction head (binary classification) - only if phase_classes is not empty
        if len(self.phase_classes) > 0:
            self.phase_head = nn.Sequential(
                nn.Linear(lstm_output_size,  hidden_size*2),
                nn.LeakyReLU(),
                nn.Linear(hidden_size*2,  hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, 2),
            )  # 2 classes for phase
        else:
            self.phase_head = None
        
        # Class prediction head (num_class classes)
        self.class_head = nn.Sequential(
            nn.Linear(lstm_output_size,  hidden_size*2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size*2,  hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, num_class),
            nn.Softmax(dim=-1)
        )
        
        self.input_dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.input_dropout(x)
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last time step output
        last_hidden = lstm_out[:, -1, :]
        
        # Get predictions from heads
        if self.phase_head is not None:
            phase_logits = self.phase_head(last_hidden)
        else:
            # Create dummy phase logits for compatibility
            phase_logits = torch.zeros(x.size(0), 2, device=x.device)
        
        class_logits = self.class_head(last_hidden)
        
        return phase_logits, class_logits
    
    def training_step(self, batch, batch_idx):
        # Ensure we're in training mode
        self.train()
        
        x, phase_target, class_target = batch
        
        # Move targets to the correct device
        phase_target = phase_target.float().to(self.device)
        class_target = class_target.to(self.device)
        
        # Apply jittering augmentation during training
        if self.training:
            noise = torch.randn_like(x) * 5
            x = x + noise
        
        # Forward pass
        phase_logits, class_logits = self(x)
        
        # Calculate losses
        if len(self.phase_classes) > 0:
            phase_loss = nn.BCEWithLogitsLoss()(phase_logits, phase_target)
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
        # Ensure we're in eval mode
        self.eval()
        
        with torch.no_grad():
            x, phase_target, class_target = batch
            
            # Move targets to the correct device
            phase_target = phase_target.float().to(self.device)
            class_target = class_target.to(self.device)
            
            # Forward pass
            phase_logits, class_logits = self(x)
            
            # Calculate losses
            if len(self.phase_classes) > 0:
                phase_loss = nn.BCEWithLogitsLoss()(phase_logits, phase_target)
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