# Telecom Churn Prediction Analysis


This project aims to predict customer churn for a telecommunications company using machine learning techniques. Churn prediction is crucial for businesses to identify customers who are likely to leave, allowing proactive measures to retain them. In this project, we explore various feature selection methods and classification models to build an accurate churn prediction model.
![Telecom churn image](telecom.png)

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
The dataset used for churn prediction contains information about telecom customers, including various features that might influence their likelihood of churning.
Features and Target Variable:
- Features: The dataset includes a range of features such as call usage, recharge patterns, customer demographics, and account information.
-	Target Variable: The target variable is typically binary, indicating whether a customer has churned or not.
Data Cleaning Steps:
-	Handling Missing Values: Missing values in the dataset were addressed through techniques such as imputation (using mean, median, or mode), deletion of rows or columns with missing values, or advanced imputation methods like K-nearest neighbors (KNN).
-	Outlier Detection and Treatment: Outliers, if present, were identified and treated using methods such as trimming, winsorization, or removing extreme values.
-	Data Scaling: Numerical features might have been scaled using techniques like standardization or normalization to ensure that they are on the same scale.
Handling Categorical Variables:
-	Encoding Categorical Variables: Categorical variables were encoded into numerical format using techniques like one-hot encoding or label encoding. This ensures that categorical variables can be used as input features in machine learning models.
-	Handling Ordinal Variables: If categorical variables had an ordinal nature, they might have been encoded with integer values preserving their order.
Data Imbalance:
-	Addressing Class Imbalance: Class imbalance between churned and non-churned customers was handled using techniques such as oversampling (e.g., SMOTE) or undersampling to balance the distribution of the target variable in the dataset.

<br>
The data preparation process involved cleaning, preprocessing, and transforming the dataset to make it suitable for building machine learning models. This included handling missing values, encoding categorical variables, addressing class imbalance, and ensuring that the dataset is ready for modeling purposes.



## Exploratory Data Analysis (EDA) <a name="exploratory-data-analysis-eda"></a>
**Data Distribution:**
-	Univariate Analysis: Histograms, box plots, and density plots were used to visualize the distribution of individual features in the dataset. This helped understand the range, central tendency, and spread of each feature.
-	Target Variable Distribution: The distribution of the target variable (churned or not churned) was examined to understand the class imbalance, which is crucial for building predictive models.
Correlation Analysis:
-	Correlation Heatmap: A correlation heatmap was generated to visualize the pairwise correlations between numerical features. This helped identify highly correlated features, which might indicate redundancy or multicollinearity.
-	Pairplot: A pairplot was used to visualize scatterplots of numerical features against each other and histograms of each feature's distribution. This provided insights into the relationship between different features.
**Feature Importance:**
-	Feature Importance Plot: Techniques like random forest or gradient boosting were used to calculate feature importance scores. A plot displaying these scores helped identify the most influential features for predicting churn.
**Patterns Observed:**
-	Churn Patterns Over Time: Time-series analysis or trend plots were used to analyze churn patterns over different time periods, such as months or quarters. This helped identify seasonal trends or changes in churn behavior over time.
-	Segmentation Analysis: Customers were segmented based on demographic or behavioral characteristics, and churn rates were compared across segments. This helped identify high-risk customer segments and tailor retention strategies accordingly.
-	Usage Patterns: Analysis of usage patterns, such as call duration, data usage, recharge frequency, etc., revealed insights into customer behavior and preferences. Differences in usage patterns between churned and non-churned customers were explored to understand factors influencing churn.

**Insights Gained**:
-	Identification of important features influencing churn behavior.
-	Understanding of correlations between features and potential multicollinearity issues.
-	Recognition of patterns and trends in churn behavior over time and across different customer segments.
-	Insights into customer behavior and preferences based on usage patterns.

The exploratory data analysis provided valuable insights into the dataset, helping understand the distribution of features, correlations between variables, and patterns in churn behavior. These insights informed subsequent steps in the data preprocessing and modeling phases, guiding feature selection, model selection, and the development of churn prediction strategies.




## Feature Engineering <a name="feature-engineering"></a>
- Feature Scaling: Standardization was performed using the StandardScaler from scikit-learn. This technique scales numerical features to have a mean of 0 and a standard deviation of 1. It helps in bringing all features to a similar scale, which is important for algorithms that are sensitive to feature scaling, such as logistic regression.
- Transformation: Log transformation was applied to skewed numerical features. Skewed features have a non-normal distribution, which can adversely affect the performance of certain models. Log transformation helps to make the distribution of these features more symmetrical, which can improve model performance.
- Creation of New Features: Interaction terms were created by combining relevant pairs of features. Interaction terms capture the combined effect of two or more features on the target variable and can provide additional predictive power to the model. Additionally, derived features were generated from existing ones to extract more meaningful information. These derived features may include ratios, differences, or aggregates of existing features.
- Handling Imbalanced Data: The dataset likely had class imbalance, where the number of instances of one class (e.g., churned customers) was significantly lower than the other class (e.g., non-churned customers). To address this imbalance, oversampling using SMOTE (Synthetic Minority Over-sampling Technique) was performed. SMOTE generates synthetic samples of the minority class by interpolating between existing minority class samples in the feature space. This helps to balance the class distribution and prevent the model from being biased towards the majority class.

By implementing these feature engineering techniques, the goal was to preprocess the data in a way that enhances the predictive power of the model and improves its ability to accurately identify churned customers.


## Model Building <a name="model-building"></a>
several machine learning models were built for churn prediction. Here's an overview of the model building process:
- Logistic Regression Model: Logistic regression is a commonly used algorithm for binary classification tasks. In this code, a logistic regression model was trained using the features obtained after feature engineering. The model was then evaluated on the test dataset to assess its performance in predicting churn.
- Principal Component Analysis (PCA) with Logistic Regression: PCA is a dimensionality reduction technique that projects high-dimensional data onto a lower-dimensional subspace while preserving the maximum amount of variance. In the code, PCA was applied to reduce the dimensionality of the feature space. A logistic regression model was then trained on the reduced feature space obtained from PCA.
- LASSO with Decision Tree and Random Forest: LASSO (Least Absolute Shrinkage and Selection Operator) is a feature selection technique that penalizes the absolute size of the coefficients, resulting in some coefficients being exactly zero. In the code, LASSO was used to select important features, which were then used to train decision tree and random forest models.
- Decision Tree Classifier: Decision trees are non-parametric supervised learning models used for classification tasks. In the code, a decision tree classifier was trained using the selected features obtained from LASSO. The model was evaluated on the test dataset to assess its performance.
- Random Forest Classifier: Random forests are an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification tasks. In the code, a random forest classifier was trained using the selected features obtained from LASSO. Similar to the decision tree model, the random forest model was evaluated on the test dataset.

Each model's performance was evaluated using metrics such as accuracy, precision, recall, and F1-score. Additionally, confusion matrices were generated to visualize the model's performance in predicting churn and non-churn instances. By building and evaluating multiple models, the goal was to identify the most effective approach for predicting customer churn in the given dataset.


### Train-Test Splitting <a name="train-test-splitting"></a>
The dataset is split into training and testing sets using the train_test_split function from the sklearn.model_selection module. A split ratio of 70% training data and 30% testing data is chosen to ensure robust evaluation of the model's performance.


### Feature Selection Techniques <a name="feature-selection-techniques"></a>
1.	Recursive Feature Elimination (RFE):
- Method: RFE is a technique that recursively removes less important features from the dataset based on their importance ranking obtained from an external estimator, such as logistic regression.
- Implementation: Logistic regression was used as the estimator, and RFE was applied to select the top features based on their importance scores.
- Result: The selected features were then used to train a logistic regression model for churn prediction.
2.	Principal Component Analysis (PCA):
- Method: PCA is a dimensionality reduction technique that transforms the original features into a new set of orthogonal variables (principal components) while retaining most of the variance in the data.
- Implementation: PCA was applied to the training data to identify the principal components that capture the maximum variance in the dataset.
- Result: The transformed principal components were used as features to train a logistic regression model or other classifiers.
3.	LASSO (Least Absolute Shrinkage and Selection Operator):
- Method: LASSO is a regularization technique that imposes a penalty on the absolute size of the coefficients, forcing some of them to shrink to zero. This results in sparse feature sets with only the most important features retained.
- Implementation: Linear Support Vector Classification (LinearSVC) with L1 regularization was used to fit the data and select important features.
- Result: Features with non-zero coefficients obtained from LASSO were considered important for predicting churn and were used for subsequent model training.

These feature selection techniques aim to reduce the dimensionality of the feature space, mitigate the curse of dimensionality, and improve model performance by focusing on the most informative features while discarding redundant or less relevant ones.


### Model Evaluation <a name="model-evaluation"></a>
Various evaluation metrics, including accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and AUC score, are used to assess the performance of each model. Insights gained from model evaluation guide the selection of the best-performing model for churn prediction.
| Model           | Accuracy | Precision (Churn) | Precision (Non-Churn) | Recall (Churn) | Recall (Non-Churn) | F1 Score (Churn) | F1 Score (Non-Churn) |
|-----------------|----------|--------------------|-----------------------|----------------|-------------------|-------------------|----------------------|
| Decision Tree   | 0.86     | 0.25               | 0.98                  | 0.74           | 0.87              | 0.38              | 0.92                 |
| Random Forest   | 0.95     | 0.51               | 0.97                  | 0.53           | 0.97              | 0.53              | 0.97                 |


## Conclusion <a name="conclusion"></a>
- The Random Forest model with LASSO feature selection demonstrated superior performance in predicting customer churn, with an accuracy of around 94% and F1-score of 0.53 for churn class and 0.97 for non-churn class.
- The results highlight the importance of feature engineering and selection in improving model performance and guiding business strategies for customer retention.
- Further research could explore additional data sources, refine feature engineering techniques, and evaluate alternative algorithms to build more robust churn prediction models.


## Future Work <a name="future-work"></a>
Future work could focus on exploring additional data sources, refining feature engineering techniques, and evaluating alternative algorithms to further improve churn prediction accuracy.

## References <a name="references"></a>
Include citations to relevant papers, articles, or documentation referenced during the project.



