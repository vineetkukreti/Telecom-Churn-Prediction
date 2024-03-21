# Telecom Churn Prediction Analysis

This repository contains code for performing churn prediction analysis on telecom customer data. The analysis includes exploratory data analysis (EDA), feature reduction using various techniques like Recursive Feature Elimination (RFE), Principal Component Analysis (PCA), and Least Absolute Shrinkage and Selection Operator (LASSO). Additionally, it includes training machine learning models such as Logistic Regression, Decision Tree, and Random Forest for churn prediction.

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

Based on the analysis, LASSO with Random Forest emerges as the most effective model for telecom churn prediction. It offers a high accuracy rate and balanced performance in predicting churn and non-churn instances.

For more details, refer to the Jupyter Notebook containing the code and analysis.

