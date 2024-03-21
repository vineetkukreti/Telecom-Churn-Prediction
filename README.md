# Telecom Churn Prediction Analysis

This repository contains code for performing churn prediction analysis on telecom customer data. The analysis includes exploratory data analysis (EDA), feature reduction using various techniques like Recursive Feature Elimination (RFE), Principal Component Analysis (PCA), and Least Absolute Shrinkage and Selection Operator (LASSO). Additionally, it includes training machine learning models such as Logistic Regression, Decision Tree, and Random Forest for churn prediction.

![Telecom churn image](telecom.png)


## Overview

The analysis is divided into the following sections:

1. Exploratory Data Analysis (EDA):
    - Scatter plot visualization for bivariate analysis.
    - Correlation matrix to identify highly correlated features.

2. Feature Reduction using RFE and Logistic Regression:
    - Implementing Recursive Feature Elimination (RFE) with Logistic Regression.
    - Balancing the dataset using oversampling (SMOTE).
    - Training a Logistic Regression model with selected features from RFE.
    - Evaluating the model performance and analyzing the confusion matrix.

3. Feature Reduction using PCA and Logistic Regression:
    - Implementing Principal Component Analysis (PCA) for dimensionality reduction.
    - Training a Logistic Regression model with PCA-transformed features.
    - Evaluating the model performance and analyzing the confusion matrix.

4. Feature Reduction using LASSO with Decision Tree and Random Forest:
    - Implementing LASSO for feature selection.
    - Training Decision Tree and Random Forest classifiers with selected features.
    - Evaluating the model performances and analyzing the confusion matrices.

## Results and Discussion

- **RFE with Logistic Regression**: Achieved an accuracy of approximately 81%, but with a higher number of false positives.
- **PCA with Logistic Regression**: Obtained an accuracy of around 82.3% with PCA-transformed features, slightly improved from RFE.
- **LASSO with Decision Tree**: Achieved an accuracy of around 86%, with a better balance between precision and recall for churn prediction.
- **LASSO with Random Forest**: Demonstrated the best performance with an accuracy of 94%, providing a higher F1 score for churn prediction and non-churn instances.

## Conclusion

Above we worked with RFE, PCA and LASSO and got following results:
1.	Using RFE we got accuracy around 81%, f1 score 0.32 ( churn ) and 0.89( non-churn ).
2.	 Using all PCA variables, accuracy was seen 82.3%, f1 score 0.34(churn), 0.90(non-churn). Usingh selected 50 variables (90% explained variance) we got almost same accuracy and f1 score. 
3.	LASSO with 

     a.	DECISSION TREE:  we got accuracy around 86%, f1 score 0.38(churn) and 0.92(non-churn)

     b.	 RANDOM FOREST:  accuracy around 94%, f1 score 0.53(churn) and 0.97(non-churn) were found.

So Lasso with Random Forest shows much better result and this model is most acceptable among above mentioned trials.

** PCA without "Balanceing data set by oversampling" was done but that shows  results:
accuracy 82.3%, f1 score 0.22(churn), 0.97(non-churn). Using selected 50 variables (90% explained variance) we got accuracy 61%, f1 score 0.15(churn), 0.74(non-churn) . This result is more poor and that  portion of programme is not shown in this notebook.





# Churn Prediction Project

## Overview
This project aims to predict customer churn for a telecommunications company using machine learning techniques. Churn prediction is crucial for businesses to identify customers who are likely to leave, allowing proactive measures to retain them. In this project, we explore various feature selection methods and classification models to build an accurate churn prediction model.

## Table of Contents
1. Introduction
2. Data Preparation
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Building
    - Train-Test Splitting
    - Feature Selection Techniques
        - Recursive Feature Elimination (RFE)
        - Principal Component Analysis (PCA)
        - LASSO (Least Absolute Shrinkage and Selection Operator)
    - Model Evaluation
6. Conclusion
7. Future Work
8. References

## Introduction
Start with an overview of the project, its objectives, and why churn prediction is important for the telecommunications industry. Discuss the significance of the project in terms of business impact and potential benefits.

## Data Preparation
The dataset used for churn prediction contains information about telecom customers, including various features that might influence their likelihood of churning.

### Features and Target Variable:
- Features: The dataset includes a range of features such as call usage, recharge patterns, customer demographics, and account information.
- Target Variable: The target variable is typically binary, indicating whether a customer has churned or not.

### Data Cleaning Steps:
- Handling Missing Values: 
    - Missing values in the dataset were addressed through techniques such as imputation (using mean, median, or mode), deletion of rows or columns with missing values, or advanced imputation methods like K-nearest neighbors (KNN).
- Outlier Detection and Treatment: 
    - Outliers, if present, were identified and treated using methods such as trimming, winsorization, or removing extreme values.
- Data Scaling: 
    - Numerical features might have been scaled using techniques like standardization or normalization to ensure that they are on the same scale.

### Handling Categorical Variables:
- Encoding Categorical Variables: 
    - Categorical variables were encoded into numerical format using techniques like one-hot encoding or label encoding.
- Handling Ordinal Variables: 
    - If categorical variables had an ordinal nature, they might have been encoded with integer values preserving their order.

### Data Imbalance:
- Addressing Class Imbalance: 
    - Class imbalance between churned and non-churned customers was handled using techniques such as oversampling (e.g., SMOTE) or undersampling to balance the distribution of the target variable in the dataset.

The data preparation process involved cleaning, preprocessing, and transforming the dataset to make it suitable for building machine learning models. This included handling missing values, encoding categorical variables, addressing class imbalance, and ensuring that the dataset is ready for modeling purposes.

## Exploratory Data Analysis (EDA)
Data Distribution:
- Univariate Analysis: Histograms, box plots, and density plots were used to visualize the distribution of individual features in the dataset.
- Target Variable Distribution: The distribution of the target variable (churned or not churned) was examined to understand the class imbalance.

Correlation Analysis:
- Correlation Heatmap: A correlation heatmap was generated to visualize the pairwise correlations between numerical features.
- Pairplot: A pairplot was used to visualize scatterplots of numerical features against each other and histograms of each feature's distribution.

Feature Importance:
- Feature Importance Plot: Techniques like random forest or gradient boosting were used to calculate feature importance scores.

Patterns Observed:
- Churn Patterns Over Time
- Segmentation Analysis
- Usage Patterns

Insights Gained:
- Identification of important features influencing churn behavior.
- Understanding of correlations between features and potential multicollinearity issues.
- Recognition of patterns and trends in churn behavior over time and across different customer segments.
- Insights into customer behavior and preferences based on usage patterns.

## Feature Engineering
- Feature Scaling
- Transformation
- Creation of New Features
- Handling Imbalanced Data

## Model Building
Several machine learning models were built for churn prediction.

### 1. Logistic Regression Model
### 2. Principal Component Analysis (PCA) with Logistic Regression
### 3. LASSO with Decision Tree and Random Forest
### 4. Decision Tree Classifier
### 5. Random Forest Classifier

## Train-Test Splitting
The dataset was split into training and testing sets using the train_test_split function from the sklearn.model_selection module.

## Feature Selection Techniques
### 1. Recursive Feature Elimination (RFE)
### 2. Principal Component Analysis (PCA)
### 3. LASSO (Least Absolute Shrinkage and Selection Operator)

## Model Evaluation
Various evaluation metrics were used to assess the performance of each model.

### Conclusion
- The Random Forest model with LASSO feature selection demonstrated superior performance in predicting customer churn.
- The results highlight the importance of feature engineering and selection in improving model performance and guiding business strategies for customer retention.

### Future Work
- Further research could explore additional data sources, refine feature engineering techniques, and evaluate alternative algorithms to build more robust churn prediction models.

### References
Include citations to relevant papers, articles, or documentation referenced during the project.
