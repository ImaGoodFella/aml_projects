# Project 1

## Task Description

"This task is primarily concerned with regression. However, we have perturbed the original MRI features in several ways. You will need to perform outlier detection, feature selection, and other preprocessing to achieve the best result."

## Dataset

This is a supervised learning task. We are given a training dataset (X_train, y_train) and a test dataset (X_test). The objective is to determine the corresponding y_test for the X_test dataset. Due to privacy concerns, I am not allowed to upload the datasets. My code assumes that they are in the current folder.

## Preprocessing

Preprocessing:

1. Outlier Detection: We tried multiple models from pyod and sklearn, but decided to remove the outliers with ECOD (about 1% of the data).

2. Feature selection: Remove columns that have a high correlation with each other. Furthermore, we removed the columns correlating with the y values using selectKbest from sklearn with the f_regressior score function. We remove columns with constant values as well.

3. Imputation: Median imputation performed best overall.

## Models

Kernels: Whenever a model required a kernel, we used the RationalKernel. The parameters for the kernel were determined using GridSearchCV.
Models: We used the sklearn StackingRegressor with the CatBoost, LightGBM, Support Vector, Relevance Vector, GaussianProcess, ExtraTrees, and Cubist regressors. The final layer of the StackingRegressor is RidgeCV.

## Comments

The local cross-validation score was much lower than the score on the leaderboard. Removing more outliers led to a higher local score but to a lower score on the leaderboard. This, combined with the lack of a description for the dataset, led to a black-box-like experience when working on the project. Consequently, most of the time was spent researching, trying out, and finally tuning different python libraries for regression models and preprocessing. In the end, we used the same preprocessing for all models. We chose the models for our stacking regressor to maximize the public score on the leaderboard. 
