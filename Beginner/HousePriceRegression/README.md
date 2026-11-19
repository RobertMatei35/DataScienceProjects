# House Price Regression

## 1. Project description

This notebook explores the Kaggle House Prices competition by building regression models to estimate a home's sale price (`SalePrice`) from its property features. It progresses from models using selected numerical features to models that preprocess both numerical and categorical data, then compares their performance.

## 2. Data description

The data comes from Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition*. It describes homes in Ames, Iowa. The training data includes the target sale price; the competition test data contains property features without sale prices.
*For this notebook data was downloaded locally outside the notebook

Features include numerical values such as lot area, overall quality and condition, year built, floor area, and room counts. Categorical features include neighborhood, lot shape, building and roof styles, foundation, heating, and exterior quality or condition.

The notebook loads `Data/train.csv` and `Data/test.csv`. Rather than imputing missing values, it removes columns containing missing values and then aligns the remaining columns between the two files.

## 3. Machine-learning workflow

1. **Load and inspect the data:** Read the training and test files, review their contents, and remove columns with missing values.
2. **Explore and select numerical features:** Use a correlation heatmap and mutual information scores to examine relationships with `SalePrice` and select a set of numerical predictors.
3. **Establish numerical baselines:** Train a standardized linear regression model and a degree-2 polynomial regression model using a train/test split.
4. **Preprocess mixed data:** Scale numerical features and one-hot encode categorical features using a scikit-learn `ColumnTransformer`.
5. **Compare models:** Fit linear regression, random forest, and gradient boosting regressors on an 80/20 train/test split. Compare predictions with actual sale prices using RMSE, R², and prediction plots.

## 4. Result overview

The notebook reports these results:

| Model | Features and preprocessing | RMSE | R² |
|---|---|---:|---:|
| Linear regression | Selected numerical features | Not reported | 0.792 |
| Polynomial linear regression (degree 2) | Selected numerical features | Not reported | 0.757 |
| Linear regression | Numerical and categorical features | $30,516.57 | 0.879 |
| Random forest | Numerical and categorical features | $30,728.98 | 0.877 |
| Gradient boosting regressor | Numerical and categorical features | $26,735.41 | 0.907 |

Gradient boosting achieved the lowest RMSE and highest R² among the models compared on the notebook's holdout split. These are local validation results, not a Kaggle leaderboard score; the notebook does not generate a competition submission.

## 5. Conclusions

Using both numerical and categorical features improved the results compared with the initial numerical-only baselines. Gradient boosting performed best in this experiment, but the result comes from a single holdout split and should not be treated as a final measure of generalization.

For future improvements, feature selection should be performed using only the training folds, missing values could be handled rather than dropping affected columns, and cross-validation and hyperparameter tuning could provide a more reliable comparison.
