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
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Model Building](#model-building)
   - [Train-Test Splitting](#train-test-splitting)
   - [Feature Selection Techniques](#feature-selection-techniques)
   - [Model Evaluation](#model-evaluation)
6. [Conclusion](#conclusion)
7. [Future Work](#future-work)
8. [References](#references)

## Introduction <a name="introduction"></a>
This project focuses on predicting customer churn for a telecommunications company using machine learning techniques. The objective is to develop an accurate churn prediction model that can help the company identify customers at risk of leaving and take proactive measures to retain them. Customer churn prediction is essential for maintaining customer satisfaction, reducing revenue loss, and maximizing profitability in the telecommunications industry.

## Data Preparation <a name="data-preparation"></a>
The dataset used for churn prediction contains information about telecom customers, including various features that might influence their likelihood of churning. Data preparation involves handling missing values, outlier detection and treatment, data scaling, and encoding categorical variables. Additionally, techniques such as addressing class imbalance are employed to ensure the dataset is suitable for building machine learning models.

## Exploratory Data Analysis (EDA) <a name="exploratory-data-analysis-eda"></a>
Exploratory Data Analysis (EDA) involves analyzing the distribution of features, correlation between variables, and patterns in churn behavior. Techniques such as univariate analysis, target variable distribution analysis, and correlation analysis are employed to gain insights into the dataset. EDA helps in understanding the underlying structure of the data and identifying key factors influencing churn behavior.

## Feature Engineering <a name="feature-engineering"></a>
Feature engineering involves transforming raw data into a format suitable for machine learning models. Techniques such as feature scaling, transformation, creation of new features, and handling imbalanced data are applied to preprocess the data. The goal is to enhance the predictive power of the model and improve its ability to accurately identify churned customers.

## Model Building <a name="model-building"></a>
Several machine learning models are built for churn prediction, including logistic regression, PCA with logistic regression, LASSO with decision tree and random forest, decision tree classifier, and random forest classifier. Each model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

### Train-Test Splitting <a name="train-test-splitting"></a>
The dataset is split into training and testing sets using the train_test_split function from the sklearn.model_selection module. A split ratio of 70% training data and 30% testing data is chosen to ensure robust evaluation of the model's performance.

### Feature Selection Techniques <a name="feature-selection-techniques"></a>
Feature selection techniques such as recursive feature elimination (RFE), principal component analysis (PCA), and LASSO are employed to reduce the dimensionality of the feature space and improve model performance.

### Model Evaluation <a name="model-evaluation"></a>
Various evaluation metrics, including accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and AUC score, are used to assess the performance of each model. Insights gained from model evaluation guide the selection of the best-performing model for churn prediction.

## Conclusion <a name="conclusion"></a>
The Random Forest model with LASSO feature selection demonstrates superior performance in predicting customer churn, highlighting the importance of feature engineering and selection in improving model performance. Further research could explore additional data sources and refine feature engineering techniques to build more robust churn prediction models.

## Future Work <a name="future-work"></a>
Future work could focus on exploring additional data sources, refining feature engineering techniques, and evaluating alternative algorithms to further improve churn prediction accuracy.

## References <a name="references"></a>
Include citations to relevant papers, articles, or documentation referenced during the project.

