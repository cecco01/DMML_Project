# DMML_Project Overview

This repository focuses on forecasting hotel booking cancellations through supervised machine learning. Leveraging historical reservation records, the solution helps hotels anticipate cancellations, improve revenue management, and streamline operations.

The analysis covers two hotel categories — a resort and an urban hotel — and follows a typical data mining pipeline (feature engineering, preprocessing, model selection and evaluation) to estimate cancellation likelihood and surface the main factors influencing cancellations.
Dataset

Data span from July 2015 to August 2017 and include bookings for two hotel types:

    - Resort Hotel (H1): 40,060 records, ~27.8% cancellations
    - City Hotel (H2): 79,330 records, ~41.7% cancellations

The dataset contains 31 attributes capturing reservation details, guest demographics and hotel-specific information (room category, special requests, etc.).

Key features

    - LeadTime: Number of days between booking date and arrival.
    - ADRThirdQuartileDeviation: How the Average Daily Rate (ADR) differs from the third quartile among comparable bookings.
    - DepositType: Deposit policy (e.g., non-refundable, none).
    - TotalOfSpecialRequests: Count of special requests made by the guest.

Preprocessing & Feature Engineering

    - Missing values were treated and non-informative columns removed.
    - Categorical encoding:
        - One-hot encoding for low-cardinality categorical features.
        - Logit-odds (or similar target-based) encoding for high-cardinality fields such as `Agent` and `Company`.
    - Class imbalance was addressed using SMOTE (Synthetic Minority Over-sampling Technique).

Tested models

I evaluated multiple classifiers, including:

    - Random Forest
    - Logistic Regression
    - AdaBoost
    - Decision Tree
    - K-Nearest Neighbors
    - XGBoost
    - Bagging
    - Naive Bayes

Model evaluation

Models were compared using metrics such as accuracy, precision, recall, F1-score and ROC AUC. The Random Forest classifier achieved the best overall results:

    - Accuracy: 0.8503
    - Precision: 0.8150
    - Recall: 0.7710
    - F1 Score: 0.7924
    - ROC AUC: 0.9119

Hyperparameter tuning

GridSearchCV was used to tune the Random Forest with a recall-focused objective. Best parameters found included:

    - `n_estimators`: 200
    - `max_depth`: None
    - `max_features`: `sqrt`

User interface

A simple GUI was built to let hotel staff enter booking information and receive an immediate cancellation probability. The UI is designed for non-technical users to support daily operational decisions.

Conclusion

This project shows that machine learning can effectively predict booking cancellations and support hotels in inventory and revenue optimization. A tuned Random Forest model produced the strongest performance in our experiments.

References

    - Antonio, Almeida, and Nunes (2017): Predicting Hotel Booking Cancellations using Machine Learning.
    - Andriawan et al. (2020): Prediction of Hotel Booking Cancellation using CRISP-DM.
    - Nuno Antonio, Ana de Almeida, Luis Nunes (2019): Hotel booking demand datasets.
