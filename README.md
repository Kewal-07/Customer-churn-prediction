
# Customer Churn Analysis - EDA and Modeling

## 📌 Overview
This project focuses on **Exploratory Data Analysis (EDA)** and predictive modeling to understand and predict **Customer Churn**.

Customer churn, also known as customer attrition, refers to the phenomenon where customers stop doing business with a company or service. It is a critical metric for businesses as it directly impacts revenue and profitability. High churn rates can often indicate dissatisfaction with the product, service, or customer experience.

The goal of this project is to uncover insights from customer data and build robust machine learning models to identify at-risk customers.

## 📂 Dataset
The dataset used in this project contains customer details and their churn status (`Exited`).

**Target Variable:**
* `Exited`: Whether the customer left the bank (1) or stayed (0).

**Features:**
* `RowNumber`, `CustomerId`, `Surname`: Identifiers (excluded from modeling).
* `CreditScore`: Customer's credit score.
* `Geography`: Customer's location.
* `Gender`: Customer's gender.
* `Age`: Customer's age.
* `Tenure`: Number of years the customer has been with the bank.
* `Balance`: Account balance.
* `NumOfProducts`: Number of bank products the customer uses.
* `HasCrCard`: Whether the customer has a credit card.
* `IsActiveMember`: Whether the customer is an active member.
* `EstimatedSalary`: Estimated annual salary.

## 🛠️ Requirements
The following Python libraries are required to run the analysis and models:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imblearn  # For SMOTE

```

##  Key Features

### 1. Handling Imbalanced Data

The dataset is processed to handle class imbalance (since fewer people churn than stay). Techniques such as **SMOTE (Synthetic Minority Over-sampling Technique)** and **Class Weighting** are implemented to ensure accurate predictions, preventing the models from being biased toward the majority class.

### 2. Exploratory Data Analysis (EDA)

Comprehensive EDA is performed to examine the data closely, identify trends, and understand the core reasons behind customer churn before modeling begins.

### 3. Classification Modeling

A variety of machine learning models were trained and evaluated:

* Logistic Regression
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* XGBoost
* Gradient Boosting

##  Results

### Model Performance Metrics

| Model | Accuracy | Recall Score | F1 Score | ROC AUC Score |
| --- | --- | --- | --- | --- |
| **Gradient Boosting** | 0.8170 | **0.7003** | **0.5984** | **0.8598** |
| XGBoost | 0.8330 | 0.6096 | 0.5870 | 0.8418 |
| Random Forest | **0.8620** | 0.4144 | 0.5390 | 0.8524 |
| Support Vector Machine | 0.7857 | 0.6627 | 0.5462 | 0.8225 |
| K-Nearest Neighbors | 0.7523 | 0.6678 | 0.5121 | 0.7766 |
| Logistic Regression | 0.7037 | 0.6832 | 0.4730 | 0.7641 |

### Analysis of Results

From the classification results, we can infer the following:

* **Gradient Boosting** has the highest F1 score (**0.598**) and the highest ROC AUC score (**0.860**) among all models. This suggests it is the most effective model in balancing precision and recall, offering the best ability to distinguish between churned and non-churned customers.
* **XGBoost** also performs well, with a relatively high F1 score (0.587) and a good ROC AUC score (0.842), making it another strong candidate for this task.
* **Random Forest** achieved the highest accuracy (**0.862**) but a significantly lower F1 score compared to the boosting models. This indicates that while it is good at predicting the majority class (non-churned), it struggles to identify the minority class (churned customers) effectively.
* **SVM and KNN** show moderate performance. They outperform Logistic Regression but fall short of the boosting algorithms.
* **Logistic Regression** yielded the lowest metrics across the board, indicating it is the least effective model for capturing the complex non-linear relationships in this dataset.

### Conclusion

**Gradient Boosting** appears to be the best model for this churn prediction task, followed closely by **XGBoost**. These models successfully handle the class imbalance and provide a robust balance between precision and recall.