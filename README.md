# Parkinson's Disease Prediction using Advanced Lightweight Ensemble

## Overview

This project implements an optimized lightweight ensemble model to predict Parkinson's Disease from fMRI images. The model combines three state-of-the-art neural network architectures (MobileNetV2, ShuffleNetV2, and GhostNet) with advanced attention mechanisms, designed specifically for small, imbalanced datasets with reliable training.

## Key Features

- **Advanced Ensemble Architecture**: Combines MobileNetV2, ShuffleNetV2, and GhostNet for optimal performance
- **Attention Mechanism**: Self-attention for intelligent feature weighting
- **Test-Time Augmentation (TTA)**: Enhanced prediction reliability through multiple augmented predictions
- **Imbalanced Dataset Handling**: Automatic class weight calculation and balancing
- **Comprehensive Evaluation**: Multiple metrics, visualizations, and performance analysis
- **Optimized for Medical Imaging**: Specifically designed for fMRI brain scan analysis

## Model Architecture

### Ensemble Components

1. **MobileNetV2 Branch**
   - Pre-trained on ImageNet
   - Fine-tuned last 30 layers for better performance
   - Optimized for mobile/lightweight deployment

2. **ShuffleNetV2 Branch**
   - Custom implementation with multiple shuffle blocks
   - Efficient channel shuffling for reduced computational cost
   - Progressive feature extraction (32→64→128→256 channels)

3. **GhostNet Branch**
   - Novel ghost modules for generating features efficiently
   - Reduced computational overhead through ghost convolutions
   - Progressive scaling (24→48→96→192 channels)

### Advanced Features

- **Self-Attention Mechanism**: Automatically weights important features
- **Residual Connections**: Improved gradient flow and training stability
- **Cyclic Learning Rate**: Dynamic learning rate scheduling for optimal convergence
- **Extensive Regularization**: Dropout, batch normalization, and L2 regularization

## Dataset

The model works with fMRI brain scan images organized in two directories:
- **PD Images**: Parkinson's Disease patients
- **Control Images**: Healthy control subjects

### Data Processing
- Image resizing to 224x224 pixels (standard for pre-trained models)
- Normalization and standardization
- Advanced augmentation (flip, rotation, zoom, contrast, brightness)
- Stratified train/validation/test split (70%/15%/15%)

## Requirements

```python
tensorflow >= 2.x
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python (cv2)
```

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd parkinsons-disease-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python
   ```

3. **Prepare your dataset**
   - Organize your fMRI images in two folders:
     - `PD_new/` - Parkinson's Disease images
     - `HC_new/` - Healthy Control images
   - Update the paths in the notebook:
     ```python
     PD_DIR = "path/to/your/PD_new"
     CONTROL_DIR = "path/to/your/HC_new"
     ```

## Usage

### Running the Model

1. **For Google Colab** (recommended):
   - Upload the notebook to Google Colab
   - Mount your Google Drive containing the dataset
   - Run all cells sequentially

2. **For Local Execution**:
   - Update the directory paths to your local dataset
   - Remove the Google Drive mounting code
   - Run the notebook in Jupyter or as a Python script

### Key Functions

- `create_mobile_shuffle_ghost_ensemble()`: Creates the main ensemble model
- `tta_predict()`: Performs test-time augmentation for robust predictions
- `find_optimal_threshold()`: Automatically finds the best classification threshold
- `save_visualizations()`: Generates comprehensive evaluation plots

## Model Performance

The model provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Detailed performance for each class
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification breakdown

### Output Visualizations

The model automatically generates:
1. Confusion Matrix heatmap
2. Training metrics plots (accuracy, loss, precision, recall, AUC)
3. ROC curve with AUC score
4. Precision-Recall curve
5. Training history CSV file

## Model Architecture Details

### Parameters
- **Total Parameters**: ~6.9M (optimized for efficiency)
- **Input Size**: 224×224×3 (RGB images)
- **Batch Size**: 16 (adjustable based on hardware)
- **Image Processing**: Standardization and normalization

### Training Configuration
- **Optimizer**: Adam with cyclic learning rate (5e-5 to 1e-4)
- **Loss Function**: Binary crossentropy
- **Callbacks**: 
  - Model checkpointing (best validation AUC)
  - Early stopping (patience: 15 epochs)
  - Learning rate reduction on plateau
  - Cyclic learning rate scheduling

## File Structure

```
parkinsons-disease-prediction/
├── pd_simple.ipynb              # Main notebook with complete implementation
├── README.md                    # This file
└── results/                     # Generated results (when run)
    ├── best_model.h5           # Best model checkpoint
    ├── advanced_ensemble_model.keras  # Final saved model
    ├── confusion_matrix.png    # Confusion matrix visualization
    ├── training_metrics.png    # Training history plots
    ├── roc_curve.png          # ROC curve
    ├── precision_recall_curve.png  # Precision-recall curve
    └── training_metrics.csv   # Training history data
```

## Key Innovations

1. **Lightweight Ensemble**: Combines efficiency with performance
2. **Medical Image Optimization**: Tailored for brain scan analysis
3. **Imbalanced Data Handling**: Automatic class weighting and balancing
4. **Robust Evaluation**: Test-time augmentation and optimal threshold finding
5. **Comprehensive Visualizations**: Complete performance analysis suite

## Results Interpretation

- **High Precision**: Low false positive rate (important for medical diagnosis)
- **High Recall**: Low false negative rate (catching actual cases)
- **Balanced Performance**: Optimized for both classes despite imbalanced data
- **Robust Predictions**: TTA provides more reliable results

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Model architecture improvements
- Additional evaluation metrics
- Performance optimizations
- Bug fixes and documentation updates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{parkinsons-ensemble-2024,
  title={Advanced Lightweight Ensemble for Parkinson's Disease Prediction},
  author={Your Name},
  year={2024},
  note={GitHub repository},
  url={https://github.com/your-username/parkinsons-disease-prediction}
}
```

## Acknowledgments

- MobileNetV2 architecture from Google
- ShuffleNetV2 implementation based on original paper
- GhostNet concepts from Huawei Noah's Ark Lab
- TensorFlow/Keras framework for deep learning implementation

## Support

For questions, issues, or support, please:
1. Check the existing issues on GitHub
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Disclaimer**: This model is for research purposes only and should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.