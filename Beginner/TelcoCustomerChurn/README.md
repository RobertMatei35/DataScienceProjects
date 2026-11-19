# Telco Customer Churn

## 1. Project description

This notebook builds a classification model to predict whether a telecommunications customer will churn. It covers data preparation, exploratory analysis, feature assessment, logistic regression, and evaluation of the model's ability to identify churners.

## 2. Data description

The data comes from Kaggle's [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It contains customer profile, tenure, subscribed services, contract, billing, and payment information, with `Churn` as the target. The original data has 21 columns, including the customer ID and target. After removing rows with blank `TotalCharges`, the notebook analyzes 7,032 records.

Example features include tenure, monthly and total charges, contract type, internet service, payment method, and optional services such as online security and tech support. The cleaned target contains 5,163 non-churners and 1,869 churners (about 26.6% churn).

## 3. Machine-learning workflow

1. **Load and inspect the data:** Download the Kaggle dataset with KaggleHub; inspect column types and missing values.
2. **Prepare the data:** Drop `customerID`, normalize “no phone/internet service” values, encode binary fields and gender, label-encode `InternetService`, `Contract`, and `PaymentMethod`, and convert `TotalCharges` to numeric after removing blank entries.
3. **Explore churn:** Visualize the churn split and examine relationships with tenure and contract type. In this dataset, churn is higher among month-to-month customers and generally decreases with longer tenure.
4. **Assess features:** Use chi-squared scores for categorical features and ANOVA scores for numerical features to explore their relationship with churn.
5. **Train and tune a classifier:** Use logistic regression in a preprocessing pipeline, a stratified 80/20 train/test split, and five-fold stratified cross-validation to search over regularization values. The search uses average precision as its scoring metric.
6. **Evaluate:** Report log loss, ROC AUC, accuracy, precision, recall, and F1, and plot the confusion matrix and ROC curve.

## 4. Result overview

On the notebook's held-out test split, the model reports:

| Metric | Result |
|---|---:|
| ROC AUC | 0.835 |
| Accuracy | 0.793 |
| Precision | 0.622 |
| Recall | 0.564 |
| F1 score | 0.592 |
| Binary cross-entropy (log loss) | 0.429 |

At the default classification threshold, recall means the model identifies about 56% of actual churners and misses about 44%. Precision means about 62% of customers it flags as likely to churn actually churn in the test set.

## 5. Conclusions

The model distinguishes churners from non-churners reasonably well, but its recall leaves many churners unidentified. Whether that balance is useful depends on the business cost of missed churners versus unnecessary retention offers; adjusting the classification threshold and evaluating that trade-off would be useful next steps.

There are also opportunities to improve the modeling workflow. `InternetService`, `Contract`, and `PaymentMethod` are label-encoded before modeling, so the logistic regression pipeline treats their integer codes as ordered numeric values rather than nominal categories. Encoding the original categorical values with one-hot encoding inside the pipeline would avoid that assumption and keep preprocessing within cross-validation.
