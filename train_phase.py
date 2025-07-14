import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger
import wandb
from phase_dataset import PhasePredictionDataset
import os
from typing import Optional
from models.phase_lstm_model import PhaseLSTMNeuralSignalModel
from models.phase_transformer_model import PhaseTransformerNeuralSignalModel
from torch.utils.data import DataLoader
from utils.phase_visualization import PhasePredictionPlotCallback
from utils.configs import BaseConfig, LSTMConfig, TransformerConfig, get_lstm_config, get_transformer_config
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

class AccuracyCallback(Callback):
    def __init__(self, test_dataset, run_name="default", num_class=4, phase_classes=None):
        super().__init__()
        self.test_dataset = test_dataset
        self.run_name = run_name
        self.best_val_class_acc = 0.0
        self.best_val_phase_acc = 0.0
        self.num_class = num_class
        self.phase_classes = phase_classes or []
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Create color maps
        self.phase_colors = ['#FF9999', '#66B2FF']  # Red for phase 1, Blue for phase 2
        self.class_colors = sns.color_palette("husl", num_class)  # Colors for classes
    
    def on_train_epoch_end(self, trainer, pl_module):
        train_phase_acc = trainer.callback_metrics.get('train_phase_acc', 0)
        train_class_acc = trainer.callback_metrics.get('train_class_acc', 0)
        print(f"\nEpoch {trainer.current_epoch}")
        if len(self.phase_classes) > 0:
            print(f"Training Phase Accuracy: {train_phase_acc:.4f}")
        print(f"Training Class Accuracy: {train_class_acc:.4f}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        val_phase_acc = trainer.callback_metrics.get('val_phase_acc', 0)
        val_class_acc = trainer.callback_metrics.get('val_class_acc', 0)
        if len(self.phase_classes) > 0:
            print(f"Validation Phase Accuracy: {val_phase_acc:.4f}")
        print(f"Validation Class Accuracy: {val_class_acc:.4f}")
        
        # If validation accuracy improved, plot confusion matrix and predictions
        if val_class_acc > self.best_val_class_acc:
            self.best_val_class_acc = val_class_acc
            self._plot_predictions(trainer, pl_module)
            self._plot_confusion_matrices(trainer, pl_module)
    
    def _plot_predictions(self, trainer, pl_module):
        """Plot model predictions vs ground truth for both phase and class"""
        # Get predictions
        pl_module.eval()
        with torch.no_grad():
            phase_preds = []
            class_preds = []
            phase_targets = []
            class_targets = []
            
            for i in range(len(self.test_dataset)):
                x, phase_target, class_target = self.test_dataset[i]
                x = torch.from_numpy(x).float().unsqueeze(0).to(pl_module.device)
                phase_logits, class_logits = pl_module(x)
                
                # Get phase prediction (binary) - only if phase_classes is not empty
                if len(self.phase_classes) > 0:
                    phase_pred = torch.sigmoid(phase_logits) > 0.5
                    phase_preds.append(phase_pred[0].cpu().numpy())
                    phase_targets.append(phase_target)
                
                # Get class prediction (only for valid samples)
                if class_target != -1:
                    class_pred = torch.argmax(class_logits, dim=1)
                    class_preds.append(class_pred[0].item())
                    class_targets.append(class_target)
        
        # Create figure with subplots based on whether we have phase prediction
        if len(self.phase_classes) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot phase predictions
            n_samples = len(phase_preds)
            rect_width = 1.0 / n_samples
            
            # Plot phase predictions and ground truth
            for i in range(n_samples):
                # Phase prediction
                pred_rect = plt.Rectangle(
                    (i * rect_width, 0),
                    rect_width,
                    0.4,
                    color=self.phase_colors[phase_preds[i][0]],
                    alpha=0.6
                )
                ax1.add_patch(pred_rect)
                
                # Phase ground truth
                gt_rect = plt.Rectangle(
                    (i * rect_width, 0.5),
                    rect_width,
                    0.4,
                    color=self.phase_colors[int(phase_targets[i][0].item())],
                    alpha=0.6
                )
                ax1.add_patch(gt_rect)
            
            # Plot class predictions
            n_valid_samples = len(class_preds)
            if n_valid_samples > 0:
                rect_width = 1.0 / n_valid_samples
                
                for i in range(n_valid_samples):
                    # Class prediction
                    pred_rect = plt.Rectangle(
                        (i * rect_width, 0),
                        rect_width,
                        0.4,
                        color=self.class_colors[class_preds[i]],
                        alpha=0.6
                    )
                    ax2.add_patch(pred_rect)
                    
                    # Class ground truth
                    gt_rect = plt.Rectangle(
                        (i * rect_width, 0.5),
                        rect_width,
                        0.4,
                        color=self.class_colors[class_targets[i]],
                        alpha=0.6
                    )
                    ax2.add_patch(gt_rect)
            
            ax2.set_title('Class Predictions vs Ground Truth')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Type')
            ax2.legend(
                [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.6) for color in self.class_colors],
                [f'Class {i}' for i in range(self.num_class)],
                loc='center left',
                bbox_to_anchor=(1, 0.5)
            )
        else:
            # Classification only - single plot
            fig, ax = plt.subplots(1, 1, figsize=(15, 8))
            
            # Plot class predictions
            n_valid_samples = len(class_preds)
            if n_valid_samples > 0:
                rect_width = 1.0 / n_valid_samples
                
                for i in range(n_valid_samples):
                    # Class prediction
                    pred_rect = plt.Rectangle(
                        (i * rect_width, 0),
                        rect_width,
                        0.4,
                        color=self.class_colors[class_preds[i]],
                        alpha=0.6
                    )
                    ax.add_patch(pred_rect)
                    
                    # Class ground truth
                    gt_rect = plt.Rectangle(
                        (i * rect_width, 0.5),
                        rect_width,
                        0.4,
                        color=self.class_colors[class_targets[i]],
                        alpha=0.6
                    )
                    ax.add_patch(gt_rect)
            
            ax.set_title('Class Predictions vs Ground Truth')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Type')
            ax.legend(
                [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.6) for color in self.class_colors],
                [f'Class {i}' for i in range(self.num_class)],
                loc='center left',
                bbox_to_anchor=(1, 0.5)
            )
        
        plt.tight_layout()
        
        # Save the figure
        save_path = f'results/predictions_{self.run_name}_epoch_{trainer.current_epoch}.eps'
        plt.savefig(save_path, format='eps', bbox_inches='tight', dpi=300)
        
        # Log to wandb
        trainer.logger.experiment.log({
            "predictions_plot": wandb.Image(plt),
            "epoch": trainer.current_epoch
        })
        plt.close()
    
    def _plot_confusion_matrices(self, trainer, pl_module):
        """Plot confusion matrices for both phase and class predictions"""
        # Get predictions
        pl_module.eval()
        with torch.no_grad():
            phase_preds = []
            class_preds = []
            phase_targets = []
            class_targets = []
            
            for i in range(len(self.test_dataset)):
                x, phase_target, class_target = self.test_dataset[i]
                x = torch.from_numpy(x).float().unsqueeze(0).to(pl_module.device)
                phase_logits, class_logits = pl_module(x)
                
                # Get phase prediction - only if phase_classes is not empty
                if len(self.phase_classes) > 0:
                    phase_pred = torch.sigmoid(phase_logits) > 0.5
                    phase_preds.append(phase_pred[0].cpu().numpy())
                    phase_targets.append(phase_target)
                
                # Get class prediction (only for valid samples)
                if class_target != -1:
                    class_pred = torch.argmax(class_logits, dim=1)
                    class_preds.append(class_pred[0].item())
                    class_targets.append(class_target)
        
        # Create figure with subplots based on whether we have phase prediction
        if len(self.phase_classes) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot phase confusion matrix
            phase_cm = confusion_matrix(
                [t[0] for t in phase_targets],
                [p[0] for p in phase_preds]
            )
            sns.heatmap(phase_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            
            # Plot class confusion matrix
            if len(class_preds) > 0:
                class_cm = confusion_matrix(class_targets, class_preds)
                sns.heatmap(class_cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
                ax2.set_title('Class Confusion Matrix')
                ax2.set_xlabel('Predicted')
                ax2.set_ylabel('True')
                ax2.set_xticklabels([f'Class {i}' for i in range(self.num_class)])
                ax2.set_yticklabels([f'Class {i}' for i in range(self.num_class)])
        else:
            # Classification only - single confusion matrix
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot class confusion matrix
            if len(class_preds) > 0:
                class_cm = confusion_matrix(class_targets, class_preds)
                sns.heatmap(class_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Class Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_xticklabels([f'Class {i}' for i in range(self.num_class)])
                ax.set_yticklabels([f'Class {i}' for i in range(self.num_class)])

        plt.tight_layout()
        
        # Save the figure
        save_path = f'results/confusion_matrices_{self.run_name}_epoch_{trainer.current_epoch}.eps'
        plt.savefig(save_path, format='eps', bbox_inches='tight', dpi=300)
        
        # Log to wandb
        trainer.logger.experiment.log({
            "confusion_matrices": wandb.Image(plt),
            "epoch": trainer.current_epoch
        })
        plt.close()

def get_dataloaders(
    train_dataset,
    test_dataset,
    batch_size: int = 32,
    num_workers: int = 0
):
    """Create train and test dataloaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def train_phase_model(
    data_dir: str = "data/raw_epi_csv",
    split_type: str = "sequential",
    run_name: str = "tmp",
    config: Optional[BaseConfig] = None,
    phase_classes: list = [],
    label_encoder: LabelEncoder = None,
    model_type: str = "lstm",
    num_class: int = 4
):
    """Train a phase prediction model"""
    # Get configuration
    if config is None:
        if model_type == "lstm":
            config = get_lstm_config(data_type="raw", split_type=split_type)
        elif model_type == "transformer":
            config = get_transformer_config(data_type="raw", split_type=split_type)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Set random seeds
    pl.seed_everything(config.seed)
    
    # Create train and test datasets
    train_dataset = PhasePredictionDataset(
        data_dir=data_dir,
        window_size=config.window_size,
        stride=config.stride,
        split_type=split_type,
        train_ratio=0.8,
        seed=config.seed,
        train=True,
        phase_classes=phase_classes,
        label_encoder=label_encoder
    )
    
    # Get the label encoder from train dataset to use for test dataset
    label_encoder = train_dataset.base_dataset.label_encoder
    
    test_dataset = PhasePredictionDataset(
        data_dir=data_dir,
        window_size=config.window_size,
        stride=config.stride,
        split_type=split_type,
        train_ratio=0.8,
        seed=config.seed,
        train=False,
        phase_classes=phase_classes,
        label_encoder=label_encoder
    )
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # Initialize wandb
    if len(phase_classes) == 0:
        # Classification only
        phase_classes_str = "classification_only"
        num_class = 4  # Use all 4 classes for classification
    else:
        phase_classes_str = "_".join(map(str, phase_classes))
    
    wandb.init(
        project="neural_signal",
        name=f"phase_{model_type}_{split_type}_w{config.window_size}_s{config.stride}_phase{phase_classes_str}_{run_name}",
        config=config.__dict__
    )
    
    # Create model based on type
    if model_type == "lstm":
        model = PhaseLSTMNeuralSignalModel(
            input_size=train_dataset.base_dataset.file_data[0]['num_channels'],
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            jitter_std=config.jitter_std,
            num_class=num_class,
            phase_classes=phase_classes,
        )
    elif model_type == "transformer":
        model = PhaseTransformerNeuralSignalModel(
            input_size=train_dataset.base_dataset.file_data[0]['num_channels'],
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            jitter_std=config.jitter_std,
            nun_class=num_class,
            phase_classes=phase_classes,
        )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=f'phase_{model_type}_{split_type}_w{config.window_size}_s{config.stride}_phase{phase_classes_str}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    
    accuracy_callback = AccuracyCallback(
        test_dataset=test_dataset,
        run_name=f"phase_{model_type}_{split_type}_w{config.window_size}_s{config.stride}_phase{phase_classes_str}_{run_name}",
        num_class=num_class,
        phase_classes=phase_classes
    )
    
    prediction_plot_callback = PhasePredictionPlotCallback(
        test_dataset=test_dataset,
        num_samples=1000,
        run_name=f"phase_{model_type}_{split_type}_w{config.window_size}_s{config.stride}_phase{phase_classes_str}_{run_name}",
        num_class=num_class,
        phase_classes=phase_classes
    )
    
    # Create wandb logger
    wandb_logger = WandbLogger(
        project="neural_signal",
        name=f"phase_{model_type}_{split_type}_w{config.window_size}_s{config.stride}_phase{phase_classes_str}"
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback, accuracy_callback, prediction_plot_callback],
        logger=wandb_logger,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        deterministic=True
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    if model_type == "lstm":
        best_model = PhaseLSTMNeuralSignalModel.load_from_checkpoint(best_model_path)
    else:  # transformer
        best_model = PhaseTransformerNeuralSignalModel.load_from_checkpoint(best_model_path)
    
    # Close wandb
    wandb.finish()
    
    return best_model, train_dataset, test_dataset

if __name__ == "__main__":
    # Define phase class configurations for experiments
    phase_configs = [
        # [3],     # Only class 3 in phase 0
        [],     # Classification only (all 4 classes) - default mode
    ]
    
    # Define model types to train
    model_types = ["transformer"]
    label_encoder = None
    
    # Train models for each configuration and model type
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} models...")
        for i, phase_classes in enumerate(phase_configs):
            if len(phase_classes) == 0:
                # Classification only
                num_class = 4
                print(f"\nTraining {model_type.upper()} model for classification only (all 4 classes)...")
                phase_classes_str = "classification_only"
            else:
                num_class = 4 - len(phase_classes)
                print(f"\nTraining Phase {model_type.upper()} model with phase classes {phase_classes}...")
                phase_classes_str = "_".join(map(str, phase_classes))
            
            # Train with sequential split
            print("\nTraining with sequential split...")
            if model_type == "lstm":
                config = get_lstm_config(data_type="raw", split_type="sequential")
            else:
                config = get_transformer_config(data_type="raw", split_type="sequential")

            model, train_dataset, test_dataset = train_phase_model(
                data_dir="data/raw_epi_csv",
                split_type="sequential",
                config=config,
                run_name=f"phase_loss500_{model_type}{phase_classes_str}",
                phase_classes=phase_classes,
                model_type=model_type,
                num_class=num_class,
            )
            
            if label_encoder is None:
                label_encoder = train_dataset.base_dataset.label_encoder

            # Train with control data
            # print("\nTraining with control data...")
            # if model_type == "lstm":
            #     config = get_lstm_config(data_type="raw", split_type="sequential")
            # else:
            #     config = get_transformer_config(data_type="raw", split_type="sequential")
            
            
            # cntl_model, cntl_train_dataset, cntl_test_dataset = train_phase_model(
            #     data_dir="data/control_csv",
            #     split_type="sequential",
            #     config=config,
            #     run_name=f"phase_classonly_{phase_classes_str}_control",
            #     phase_classes=phase_classes,
            #     label_encoder=label_encoder,
            #     model_type=model_type,
            #     num_class=num_class,
            # ) 