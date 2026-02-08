# Multiclass Beef Classification using E-Nose Sensors

## Purpose
This project performs multiclass classification of beef quality using time-series data collected from an electronic nose (e-nose) sensor system.
Beef quality is classified into 4 classes: excellent, good, acceptable, spoiled

## Dataset
- **Source**: Electronic Nose Dataset for Beef Quality Monitoring
- **Files**: 5 CSV files (TS1.csv - TS5.csv)
- **Train set**: TS1.csv - TS4.csv (concatenated)
- **Test set**: TS5.csv
- **Sampling rate**: 1 minute
- **Duration**: 36 hours per experiment
- **Sensors**: MQ135, MQ136, MQ2, MQ3, MQ4, MQ5, MQ6, MQ8, MQ9
- **Extra features**: Temperature, Humidity
- **Number of classes**: 4 (excellent=0, good=1, acceptable=2, spoiled=3)
- **Target Variable**: Classification of beef

## Pipeline

### 1. Data Loading & Preparation
- Load 4 training files (TS1-TS4)
- Combine into a single DataFrame
- Load test file (TS5) separately

### 2. Data Exploration (EDA)
- Check for missing values and outliers
- Plot histograms, boxplots, and correlation matrix
- Analyze distribution by class

### 3. Label Encoding
- Encode class from text → integer (0-3)
- Order: excellent(0) → good(1) → acceptable(2) → spoiled(3)

### 4. Feature Scaling
- Standardize all sensor data to a scale (approx. -2 to +2)
- Mandatory for SVM, optional for Random Forest

### 5. Train-Test Split
- X_train, y_train: TS1-TS4 (scaled)
- X_test, y_test: TS5 (scaled using training scaler)

### 6. Model Training & Evaluation

## Models

### Random Forest (Baseline)
RandomForestClassifier(n_estimators=200, random_state=42)

Advantages:
- Not sensitive to scale
- Handles multicollinearity well
- Provides feature importance
- Less prone to overfitting (ensemble)

### SVM (Support Vector Machine)
SVC(kernel='rbf', C=0.1, gamma='scale', random_state=42, class_weight='balanced')

Advantages:
- Effective for high-dimensional data
- RBF kernel handles non-linear relationships
- class_weight='balanced' prevents class bias

## Results

### Metrics
- Accuracy: Overall correct prediction rate
- Precision: % correct among predicted class
- Recall: % of actual samples captured
- F1-score: Balance between precision & recall

### Confusion Matrix
- Diagonal: Correct predictions 
- Off-diagonal: Incorrect predictions 

### Feature Importance
- Identify which sensors are most significant

## Output Files
./predict/
├── random_forest.csv    # Results from RF
└── svm.csv              # Results from SVM

Each file contains:
- minute: Measurement time
- actual: Actual class
- pred: Predicted class
- correct: True/False (prediction accuracy)

## Getting Started

### Requirements
pip install pandas scikit-learn matplotlib seaborn numpy

### Run Notebook
jupyter notebook "Multiclass Beef Classification.ipynb"

### Execution Steps
1. Importing Data
2. Data Cleaning
3. Label Encoding
4. EDA
5. Feature Engineering
6. Model Training
7. Evaluation

## Insights & Observations

### Data Analysis
- Sensor signals are noisy and drifting
- Significant overlap between MQ5 vs MQ6
- Environmental factors (temperature & humidity) affect sensor responses
- Some sensors contribute more than others (MQ4)

### Future Improvements
- GridSearchCV for hyperparameter tuning
- Dimensionality reduction (PCA)
- Cross-validation across experiments
- Imbalanced data handling (SMOTE)
- Time-aware models (LSTM, Temporal CNN)