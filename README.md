# Customer Churn Prediction

A machine learning project that predicts which telecom customers are likely to leave. It compares nine classification models, tunes the strongest three, and selects a final Random Forest model that reaches an F1 score of 0.64 and an accuracy of 76% on unseen data.

## Table of Contents

* [Business Problem](#business-problem)
* [Dataset](#dataset)
* [Approach](#approach)
* [Results](#results)
* [Limitations](#limitations)
* [Repository Structure](#repository-structure)
* [How to Run](#how-to-run)
* [Tools Used](#tools-used)


## Business Problem

Keeping an existing customer is usually cheaper than winning a new one. If a telecom company can identify customers who are about to leave, it can target them with retention offers before they go. This project builds a classification model that flags likely churners from their account, service and billing information.

## Dataset

The project uses the IBM Telco Customer Churn dataset, available from [Mendeley Data](https://data.mendeley.com/datasets/phsxg9ssrf/1). A copy is also included in this repository as `Telco_customer_churn_dataset.csv`.

* Each row is one customer, and the columns describe their demographics, location, subscribed services, contract, billing and whether they churned.
* The dataset has 7,043 customers: 5,174 stayed and 1,869 churned, so about 27% churned.
* Because the classes are uneven, accuracy alone is misleading. Models were compared mainly on the **F1 score for the churn class**, with AUC ROC as a supporting measure.

## Approach

### 1. Data cleaning and exploration

* Converted `Total Charges` from text to a numeric column and dropped the 11 rows where it was missing.
* Confirmed that the missing values in `Churn Reason` belong to customers who did not churn.
* Summarised the key numeric fields. The average monthly charge is about $65 and the average tenure is about 32 months, with a wide spread in tenure.

### 2. Preparing the data for modelling

* Dropped identifier columns (`CustomerID`, `Count`) and the location columns that overlap with each other (`Country`, `State`, `City`, `Zip Code`, `Lat Long`), keeping latitude and longitude.
* Dropped `Churn Reason`, `Churn Label` and `Churn Score`. `Churn Reason` only exists for customers who already left, `Churn Label` duplicates the target, and `Churn Score` is a pre-calculated metric that would leak information into training.
* One hot encoded the categorical variables.
* Split the data 70% for training and 30% for testing, stratified so both sets keep the same churn rate.
* Scaled the numeric features (tenure, monthly charges, total charges, CLTV, latitude, longitude) using a scaler fitted on the training set only, then applied to the test set, to avoid data leakage.

### 3. Handling class imbalance

* Used balanced class weights for Logistic Regression, Random Forest, SVM, Decision Tree and LightGBM.
* Used a `scale_pos_weight` ratio for XGBoost.
* SMOTE was considered as an alternative but not applied.

### 4. Model comparison

Nine models were trained on the same data and evaluated on accuracy, precision, recall, F1 and AUC ROC:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Support Vector Machine
5. Gaussian Naive Bayes
6. k Nearest Neighbours
7. Decision Tree
8. XGBoost
9. LightGBM

### 5. Hyperparameter tuning

The three best models on F1 (LightGBM, Logistic Regression and Random Forest) were tuned using grid search with 5 fold cross validation on the training data, optimising for F1.

### 6. Feature importance and final model

* Ranked the features by importance using the tuned Random Forest.
* Tested removing the least important feature (`Phone Service`). It made no meaningful difference to performance, so it was kept in the final model because it is useful account information.

## Results

F1 scores for the churn class on the test set:

* **Before tuning:** LightGBM 0.629, Logistic Regression 0.622, Random Forest 0.621.
* **After tuning:** the Random Forest led the three models with an F1 score of **0.639**.
* **Final model:** the tuned Random Forest reached an accuracy of **0.76** on the test set.

The full metrics table, confusion matrix and classification report are in the notebook.

## Limitations

* The model was evaluated on a single train and test split. Repeated cross validation on the full pipeline would give a more stable estimate of performance.
* An F1 score of about 0.64 means the model still misses some churners and flags some loyal customers. It is best used to prioritise customers for review, not as a final decision.
* The data is a sample dataset from IBM, so results may not transfer directly to another company's customers.
* The model is not deployed, and there is no monitoring for drift over time.
* The model predicts who is likely to leave, not why. Feature importance shows association, not cause.

## Repository Structure

```
Customer-Churn-Machine-Learning-Application/
    Machine_Learning_-_Customer_Churn.ipynb    Full analysis and modelling notebook
    Telco_customer_churn_dataset.csv           Dataset used in the project
    README.md                                  Project overview
```

## How to Run

1. Clone the repository:

   ```
   git clone https://github.com/Glen-Ochieng/Customer-Churn-Machine-Learning-Application.git
   cd Customer-Churn-Machine-Learning-Application
   ```

2. Install the required libraries:

   ```
   pip install pandas numpy scikit-learn xgboost lightgbm jupyter
   ```

3. Open the notebook and run the cells from top to bottom:

   ```
   jupyter notebook "Machine_Learning_-_Customer_Churn.ipynb"
   ```

The notebook loads the dataset directly from this repository, so it needs an internet connection unless you change the file path to the local CSV. Hyperparameter tuning uses a lot of memory and can take several minutes. A fixed random seed (42) is used throughout so results can be reproduced.

## Tools Used

* **Language:** Python
* **Data handling:** pandas, NumPy
* **Modelling:** scikit-learn, XGBoost, LightGBM
* **Environment:** Jupyter Notebook

