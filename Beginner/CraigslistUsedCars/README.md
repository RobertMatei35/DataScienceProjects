# Craigslist Used Car Price Prediction

## 1. Project description

This notebook develops a regression model to estimate used-car listing prices using Craigslist vehicle data. The project focuses on data cleaning and exploratory analysis, handling unusual prices and mileage, engineering vehicle age, and comparing a tuned XGBoost regressor against held-out data.

## 2. Data description

The data comes from Kaggle's [Craigslist cars and trucks listings](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data). The original dataset contains 426,880 listings and 26 columns. The target is `price`; the remaining fields describe vehicle characteristics and listings, including model year, manufacturer, model, condition, cylinders, fuel, odometer, transmission, drive, body type, paint color, location, and posting date.

The notebook reads `Data/vehicles.csv`. It removes duplicate and irrelevant listing fields, drops columns with more than 50% missing values, and removes rows missing several required fields. It also filters prices to the range above $1,000 and up to $200,000, removes prices matching simple repeated-digit or sequence patterns, and excludes vehicles over 45 years old. These cleaning choices exclude call for price listings, rare cars with unique price dynamics as supercars and colletion cars.

## 3. Machine-learning workflow

1. **Load and inspect:** Review the 26 columns, data types, and missing values.
2. **Clean the listings:** Identify duplicates, remove irrelevant columns, and handle missing values by dropping columns or rows according to the notebook's rules.
3. **Explore and filter prices:** Inspect price distributions and remove low, extreme, and pattern-like prices considered suspicious. Apply a logarithmic transformation to reduce price skew.
4. **Explore and engineer features:** Examine numeric and categorical features, normalize category text, calculate vehicle age from the listing date and model year, and inspect mileage outliers.
5. **Prepare model inputs:** Use log-transformed price as the target. Scale `year`, `odometer`, and `age`, and one-hot encode `fuel` and `transmission` in a scikit-learn preprocessing pipeline. Categorical features with +5 values are maintained as they are since XGBoost can handle them.
6. **Train and tune:** Fit an XGBoost regressor and use a five-fold shuffled cross-validation grid search over 108 hyperparameter combinations, optimizing R².
7. **Evaluate:** Report R², MAE, MSE, and RMSE on a held-out test split and plot predicted versus actual log prices.

## 4. Result overview

The best cross-validation R² was **0.646**. On the held-out test split, the notebook reports:

| Metric | Result |
|---|---:|
| R² | 0.643 |
| MAE | 0.391 |
| MSE | 0.261 |
| RMSE | 0.511 |

These error metrics are calculated against the **log-transformed price target**, not in dollars. They should not be interpreted as dollar prediction errors; the notebook does not report errors after converting predictions back to the original price scale.

## 5. Conclusions

The tuned XGBoost model captures some of the variation in log listing prices, with a test R² of about 0.64. This is a useful baseline, but it does not establish how accurate individual price estimates are in dollars.

The price and vehicle-age filters limit the scope of the model, and several potentially informative fields—such as model, condition, drive, and vehicle type—are not included in the final features because of their high null values. 
Next steps will include usage of NHTSA api to retrieve missing features like model or drive, testing richer categorical encoding and additional features, and comparing results across cross-validation folds.
