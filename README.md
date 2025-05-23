# Brain Tumor Detection System

A machine learning-based system for detecting brain tumors from MRI scans using advanced feature extraction and ensemble learning techniques.

## Overview

This project implements an automated brain tumor detection system that analyzes MRI scans to identify the presence of tumors. The system uses a combination of advanced image processing techniques and machine learning algorithms to achieve high accuracy in tumor detection.

## Features

- **Advanced Image Processing**:
  - Contrast enhancement using CLAHE
  - Noise reduction using non-local means denoising
  - Standardized image preprocessing pipeline

- **Comprehensive Feature Extraction**:
  - Basic intensity features
  - Edge detection features
  - Texture analysis (GLCM)
  - Local Binary Pattern (LBP) features
  - Gabor filter features
  - Wavelet transform features

- **Robust Machine Learning Model**:
  - Stacking ensemble of Random Forest classifiers
  - Feature selection for optimal performance
  - Cross-validation for reliable results
  - High accuracy (85.75%) in tumor detection

## Technical Details

### Model Architecture
- **Base Models**: Two Random Forest classifiers with different configurations
- **Meta-Learner**: Logistic Regression
- **Feature Selection**: SelectFromModel with Random Forest
- **Cross-Validation**: 5-fold cross-validation

### Performance Metrics
- Accuracy: 85.75%
- Precision (Tumor): 0.78
- Recall (Tumor): 0.83
- F1-Score (Tumor): 0.80

## Dataset

The project uses the Kaggle 3M dataset, which includes:
- 3,929 MRI scans
- 1,373 tumor cases
- 2,556 non-tumor cases
- High-quality medical images with corresponding mask files

## Installation

This is a private project. No installation guide will be provided. Only important files are updated in the repository.

## Project Structure

```
Brain Tumor Detector/
├── Source Code/
│   ├── api.py
│   ├── tumor_detector.py
│   ├── train_model.py
│   └── myProject.ipynb
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- OpenCV
- scikit-learn
- scikit-image
- PyWavelets
- NumPy
- Pandas
- Flask (for API)

## Contributors

- Vishesh Varshney (229301689)
- Aayush Dnyane (229301638)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Cancer Genome Atlas (TCGA) for the dataset
- Kaggle for hosting the 3M dataset
- All contributors and supporters of the project

## Contact

For any queries or suggestions, please contact:
- Vishesh Varshney: [email]
- Aayush Dnyane: [email]
