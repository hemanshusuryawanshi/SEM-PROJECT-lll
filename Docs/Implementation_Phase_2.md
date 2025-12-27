# Implementation Phase – 2  
## Model Development & Training

This phase focuses on building the core intelligence of the Health Insurance Purchase Prediction System. The goal of this phase is to transform raw customer data into a trained machine learning model capable of predicting whether a customer is likely to purchase health insurance.

## 1. Dataset Understanding

The dataset used in this project contains customer-related information collected from health insurance records. Each row represents a customer, and each column represents a feature that influences insurance purchase behavior.

### Key Features:
- Age
- Gender
- Income
- Occupation
- Education Level
- BMI (Body Mass Index)
- Smoking Habit
- Family Size
- Previous Insurance History
- Region

The target variable is:
- **Purchase Decision** (Yes / No)

## 2. Data Preprocessing

Raw data cannot be directly used for machine learning. Therefore, multiple preprocessing steps were applied to improve data quality and ensure consistency.

### Steps Performed:
- Handling missing values using appropriate strategies
- Encoding categorical variables using Label Encoding
- Scaling numerical features using StandardScaler
- Removing unnecessary or redundant columns
- Ensuring correct feature order for model training

All preprocessing steps were saved as reusable artifacts to maintain consistency during prediction.

## 3. Model Selection

Three different classification algorithms were implemented and evaluated:

### a) Logistic Regression
- Used as a baseline model
- Simple and interpretable
- Works well for linear relationships

### b) Random Forest
- Ensemble-based learning method
- Handles non-linear relationships effectively
- Reduces overfitting compared to single decision trees

### c) XGBoost
- Advanced gradient boosting algorithm
- Handles missing values efficiently
- Provides high accuracy for structured/tabular data
- Captures complex feature interactions

## 4. Model Training

Each model was trained using the preprocessed dataset. The dataset was split into training and testing sets to evaluate model performance fairly.

### Evaluation Metrics Used:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Among all models, **XGBoost achieved the highest accuracy**, making it the final selected model for deployment.

## 5. Saving Model Artifacts

To ensure reproducibility and consistent predictions during deployment, the following artifacts were saved:
- Trained machine learning model (`.pkl`)
- Label encoders
- Scaler object
- Feature name list

These artifacts are later loaded during the prediction phase.

## 6. Outcome of Phase 2

At the end of this phase:
- A high-performing ML model is ready
- Preprocessing pipeline is finalized
- All artifacts are stored for deployment
- The system is prepared for real-time inference

This phase lays the foundation for deploying the prediction system in a real-world environment.
