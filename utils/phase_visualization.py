import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.data import DataLoader
import wandb
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import confusion_matrix
import os

class PhasePredictionPlotCallback(Callback):
    """Callback for plotting phase prediction results during training"""
    
    def __init__(self, test_dataset, num_samples=1000, run_name="default", num_class=4, phase_classes=None):
        super().__init__()
        self.test_dataset = test_dataset
        self.num_samples = min(num_samples, len(test_dataset))
        self.run_name = run_name
        self.num_class = num_class
        self.phase_classes = phase_classes or []
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Create color maps
        self.phase_colors = ['#FF9999', '#66B2FF']  # Red for phase 1, Blue for phase 2
        self.class_colors = sns.color_palette("husl", num_class)  # Colors for classes
        
    def on_train_epoch_end(self, trainer, pl_module):
        # Plot predictions every 10 epochs
        if (trainer.current_epoch + 1) % 10 == 0:
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
            
            for i in range(self.num_samples):
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
        save_path = f'results/phase_predictions_{self.run_name}_epoch_{trainer.current_epoch}.eps'
        plt.savefig(save_path, format='eps', bbox_inches='tight', dpi=300)
        
        # Log to wandb
        trainer.logger.experiment.log({
            "phase_predictions_plot": wandb.Image(plt),
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
        save_path = f'results/phase_confusion_matrices_{self.run_name}_epoch_{trainer.current_epoch}.eps'
        plt.savefig(save_path, format='eps', bbox_inches='tight', dpi=300)
        
        # Log to wandb
        trainer.logger.experiment.log({
            "phase_confusion_matrices": wandb.Image(plt),
            "epoch": trainer.current_epoch
        })
        plt.close() 