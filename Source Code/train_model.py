import os
import numpy as np
import cv2
from tqdm import tqdm
from tumor_detector import TumorDetector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    """
    Load and preprocess the training data
    """
    features = []
    labels = []
    
    # Get the correct path to kaggle_3m directory
    data_dir = os.path.join(os.path.dirname(__file__), 'kaggle_3m')
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at: {data_dir}")
    
    # Get all patient directories
    patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(patient_dirs)} patient directories")
    
    detector = TumorDetector()
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_path = os.path.join(data_dir, patient_dir)
        
        # Get all image files (excluding masks)
        image_files = [f for f in os.listdir(patient_path) 
                      if f.endswith('.tif') and not f.endswith('_mask.tif')]
        
        for img_file in image_files:
            img_path = os.path.join(patient_path, img_file)
            mask_path = os.path.join(patient_path, img_file.replace('.tif', '_mask.tif'))
            
            # Check if mask exists and has tumor pixels
            has_tumor = False
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None and np.any(mask > 0):
                    has_tumor = True
            
            # Extract features
            img = cv2.imread(img_path)
            if img is not None:
                img_features = detector.extract_features(img)
                if img_features is not None:
                    features.append(img_features)
                    labels.append(1 if has_tumor else 0)
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"\nLoaded {len(X)} samples")
    print(f"Number of tumor samples: {np.sum(y == 1)}")
    print(f"Number of non-tumor samples: {np.sum(y == 0)}")
    
    return X, y

def main():
    # Initialize detector with explicit paths
    detector = TumorDetector(
        model_path='./trained_model.joblib',
        scaler_path='./scaler.joblib'
    )
    
    # Check if model already exists
    if os.path.exists('./trained_model.joblib') and os.path.exists('./scaler.joblib'):
        print("Model already exists. Skipping training.")
        print("To retrain the model, delete 'trained_model.joblib' and 'scaler.joblib' files.")
        return
    
    # Load data
    print("Loading data...")
    X, y = load_data()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    print("\nTraining model...")
    detector.train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = detector.model.predict(detector.scaler.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 