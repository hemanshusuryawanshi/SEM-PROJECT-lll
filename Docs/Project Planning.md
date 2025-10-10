# Project Planning

## Project Name: Health Insurance Purchase Prediction

## Project Goal
The main goal of this project is to **predict whether a customer will buy health insurance or not**.
By using real-world data from **Kaggle**, we can understand what factors influence a customer’s decision and help insurance companies reach the right people.

## Project Steps

### 1. Understand the Problem
* Find out what we want to predict and why it’s useful.
* Identify what customer details are important for making predictions.

### 2. Collect and Explore Data
* Use the dataset from **Kaggle**.
* Look at the data carefully — check for missing values, unusual entries, and overall structure.

### 3. Clean and Prepare the Data
* Fix or remove missing and incorrect values.
* Convert text data (like gender or occupation) into numbers so the computer can understand.
* Normalize or scale number-based data like age and income.
* Split the dataset into **training data** (for learning) and **testing data** (for checking accuracy).

### 4. Build and Train the Model
* Try out different machine learning models such as:
  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting (XGBoost/LightGBM)
* Train the model to recognize patterns in the data.

### 5. Test and Evaluate
* Test how well the model works using measures like:
  * **Accuracy** (how often the model is right)
  * **Precision** and **Recall** (how well it finds true buyers)
  * **ROC-AUC** (overall model performance)
* Choose the best model based on results.

### 6. Make Predictions
* Use the final model to predict if a new customer is likely to buy health insurance or not.
