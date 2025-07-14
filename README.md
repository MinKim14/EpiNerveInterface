# EpiNerveInterface
# 🧠 Behavior Decoding with the e-Epi (Coming Soon!)

Yeonzu Son1†, Hyeonseok Jeong2†, Min Kim3, Somin Lee1, Ain Chung1,4, Jiheong Kang2*, Hyunwoo Yuk5*‡, Seongjun Park 3,6,7,8,9*

This repository will provide the official code, pretrained models, and datasets for the paper:

**"Anti-fibrotic Electronic Epineurium for High-fidelity Chronic Peripheral Nerve Interface in Freely Moving Animals"**

In this work, we demonstrate high-accuracy behavior decoding using signals recorded from our novel electronic epineurium (e-Epi) interface.

## 🚀 Key Features
- End-to-end **behavior decoding** pipeline using e-Epi signals
- Transformer-based deep learning models
- Lightweight linear models for comparison
- Dataset for freely moving animal nerve recordings
- Reproducible experiments & evaluation scripts

## 📝 Paper
[Coming Soon] 📄

## 📂 Code & Models

This repository contains the implementation of deep learning models for neural signal classification and phase prediction. The code supports both standard multi-class classification and advanced phase separation training for distinguishing between different behavioral states.

## Overview

The project implements neural signal models that can perform:
1. **Standard Classification**: Multi-class classification of neural signals (default: 4 classes)
2. **Phase Separation Training**: Advanced dual-task learning that separates certain classes (e.g., immobile states) into Phase 0 for further behavioral distinction

The models can be configured for **classification-only** training (default) or **phase separation** training when specific classes are designated as Phase 0.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Installing Dependencies

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/MinKim14/EpiNerveInterface.git
   cd EpiNerveInterface
   ```

2. **Create a virtual environment** :
  ```bash
  # Create conda environment
  conda create -n neural_signal python=3.10
  conda activate neural_signal
  
  # Install dependencies
  pip install -r requirements.txt
  ```

### PhaseTransformerNeuralSignalModel
- **Architecture**: Transformer encoder with dual prediction heads
- **Features**:
  - Multi-head self-attention
  - Positional encoding
  - Phase and class prediction heads - phase head is optional
  - Data augmentation with jittering

## Training

### Main Training Script
```bash
python train_phase.py
```

### Configuration
The training uses configuration classes defined in `utils/configs.py`:
- `LSTMConfig`: Configuration for LSTM models
- `TransformerConfig`: Configuration for Transformer models
  
When `phase_configs` is empty (`[]`):
- Only class prediction is performed
- All 4 classes are used for classification
- Phase prediction head is disabled
- Only classification plots and confusion matrices are generated

When `phase_configs` contains class indices:
- Performs both phase and class prediction
- Designated classes become Phase 0
- Remaining classes become Phase 1
- Creates two subplots for visualization
- Uses `num_class = 4 - len(phase_classes)` for class prediction

## Results
Training results are saved in:
- `results/`: Visualization plots and confusion matrices
- `checkpoints/`: Model checkpoints
- WandB logging for experiment tracking

## Data Access

To obtain the neural signal data used in this study, please contact:
**yeonzus@kaist.ac.kr**

The data includes raw neural recordings with behavioral state annotations suitable for both classification and phase separation training.
   
## 📋 Citation
If you use this repository, please cite our paper:

