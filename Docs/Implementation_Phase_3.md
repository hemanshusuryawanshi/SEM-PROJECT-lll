# Implementation Phase – 3  
## Deployment, Prediction & Database Integration

This phase focuses on deploying the trained machine learning model into a usable system. The aim is to allow users to interact with the model through a simple interface and receive real-time predictions.

## 1. Deployment Architecture

The system is deployed using a lightweight and efficient architecture consisting of:
- Streamlit for frontend interface
- Python backend for prediction logic
- SQLite database for logging predictions

This approach ensures simplicity, speed, and ease of maintenance.

## 2. Streamlit User Interface

Streamlit is used to build the complete frontend of the application.

### UI Features:
- Form-based input for customer details
- Dropdowns for categorical attributes
- Numeric input fields for age, income, BMI, etc.
- A single **Predict** button to trigger inference
- Display of prediction result and probability score
- View of recent prediction history

The interface is designed to be user-friendly so that even non-technical users can operate the system easily.

## 3. Backend Prediction Logic

Once the user submits the input form, the backend performs the following steps:

1. Collects user input from the UI
2. Applies the same preprocessing steps used during training
3. Loads the saved machine learning model
4. Generates prediction probability using `predict_proba()`
5. Applies a threshold (0.5) to classify results:
   - ≥ 0.5 → Likely to Purchase
   - < 0.5 → Not Interested

This ensures accurate and consistent predictions.

## 4. Database Integration (SQLite)

To maintain prediction records, a local SQLite database is used.

### Stored Information:
- Customer input details
- Prediction result
- Probability score
- Timestamp
- Model version (if applicable)

SQLite is chosen because it is lightweight, serverless, and ideal for academic projects and prototypes.

## 5. Error Handling & Validation

The system includes basic validation to ensure:
- No invalid or empty inputs are processed
- Unexpected categories are handled safely
- The application does not crash during runtime

This improves system reliability and user trust.

## 6. System Testing

The complete application was tested using multiple sample inputs to verify:
- Correct prediction logic
- Proper preprocessing
- Accurate probability output
- Successful database logging
- Smooth UI interaction

## 7. Outcome of Phase 3

At the end of this phase:
- The ML model is fully deployed
- Users can perform real-time predictions
- Prediction history is stored successfully
- The system works as an end-to-end solution

This phase completes the implementation of the Health Insurance Purchase Prediction System, making it ready for academic demonstration and further enhancements.
