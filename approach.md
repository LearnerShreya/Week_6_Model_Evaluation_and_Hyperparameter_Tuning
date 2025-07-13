
# Approach - Week 6 Project

This file outlines the detailed machine learning approach used to build and evaluate a survival prediction model for Titanic passengers.


## 1. Dataset and Objective

The dataset (`train.csv`) contains historical data of Titanic passengers including features like age, fare, class, sex, and whether they survived or not.  
The objective is to build models that can classify whether a passenger survived (`1`) or not (`0`).

## 2. Data Preprocessing

### a. Feature Selection
- Dropped columns not relevant to prediction: `Name`, `Ticket`, `Cabin`, `PassengerId`

### b. Handling Missing Values
- `Age`: Filled missing values with the **median**
- `Embarked`: Filled with the **mode**
- Remaining missing rows were dropped

### c. Categorical Encoding
- Converted `Sex` into binary values: `male → 0`, `female → 1`
- Converted `Embarked`: `S → 0`, `C → 1`, `Q → 2`

### d. Final Features Used:
- `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`


## 3. Model Building

### a. Models Trained (Before Tuning)
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

Each model was trained using `X_train` and evaluated on `X_test`.

## 4. Evaluation Metrics

The following classification metrics were used to compare models:
- **Accuracy**: Overall correctness
- **Precision**: True Positive rate out of predicted Positives
- **Recall**: True Positive rate out of actual Positives
- **F1 Score**: Harmonic mean of Precision and Recall
- **Confusion Matrix**: For deeper error analysis


## 5. Hyperparameter Tuning

### a. Random Forest (GridSearchCV)
Used `GridSearchCV` to tune:
- `n_estimators`, `max_depth`, `min_samples_split`

### b. Decision Tree (RandomizedSearchCV)
Used `RandomizedSearchCV` to tune:
- `max_depth`, `min_samples_split`, `criterion`

### c. Best Models After Tuning
- **best_rf** → Best-tuned Random Forest
- **best_dt** → Best-tuned Decision Tree

These were selected using `best_estimator_` from the tuning results.


## 6. Model Comparison

All models (before and after tuning) were compared based on:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix


## 7. Model Selection and Saving

- The **Tuned Random Forest** performed the best across all metrics.
- It was saved using `joblib` as: `titanic_model.pkl`

```python
import joblib
joblib.dump(best_rf, "titanic_model.pkl")
```


## 8. Deployment (Streamlit)

* A simple web application was built using **Streamlit**
* Allows users to enter passenger details and receive survival predictions in real-time
* The app loads the saved `.pkl` model and performs prediction on user input

```bash
streamlit run app.py
```

## Summary

This project demonstrates the full pipeline of:

* Data preprocessing
* Model training and comparison
* Performance tuning
* Model saving
* Web deployment