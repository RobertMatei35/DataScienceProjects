# Battery Remaining Useful Life (RUL) Estimator

## 1. Project description

This notebook explores using a Keras artificial neural network (ANN) to estimate battery aging from charge and discharge measurements. It builds cycle-level features for four NASA batteries (B0005, B0006, B0007, B0018) and trains the network to predict discharge capacity, from which State of Health (SOH) and Remaining Useful Life (RUL) are derived. The ANN is compared with simple baseline models. The work is inspired by Kumar's study of battery SOH and RUL estimation [1]. For additional context on machine-learning approaches to these tasks, see [2].

This is an exploratory modeling project, not an industry-validated RUL system. The model predicts capacity; it does not directly train on RUL as its target.

## 2. Data description

The notebook uses the [NASA Battery dataset on Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset), which provides battery cycling data in CSV form. The referenced [NASA Battery Aging dataset files](https://labinfo.ing.he-arc.ch/gitlab/ticc/16TICc19/nasa-battery-dataset/-/tree/614f3518915079a5182655cadc3492973ac90d6c/data/BatteryAgingARC-FY08Q4) contain charge, discharge, and impedance measurements across repeated cycles. The four batteries were tested under the same conditions.

Measurements include voltage, current, temperature, and time during charge and discharge operations. Discharge capacity is used as the model target. SOH is calculated relative to an initial capacity of 2 Ah. RUL is derived from the cycle's position in the battery's recorded lifetime (cycles remaining until the end of the test).

## 3. Machine-learning workflow

1. **Load and inspect the data:** Use the metadata to identify the cycle files of each battery and examine sample charge and discharge profiles.
2. **Prepare cycle pairs:** Inspect and remove duplicate operations, then pair consecutive charge and discharge records to represent complete cycles.
3. **Engineer features:** Extract charge constant-current and constant-voltage durations, charge-start voltage, charge and discharge temperature rise (max - min of the cycle), minimum discharge voltage, and the rest interval between cycles. Rest intervals become two exponentially decaying signals (unusually long and unusually short rests, relative to each battery's median, three-cycle half-life). Derive SOH and RUL from capacity and cycle order.
4. **Select model inputs:** Use eight standardized inputs to predict capacity. Discharge time is excluded because, at constant current, it is directly related to capacity. Temperature is used as a rise rather than a maximum, since the batteries start charging at different temperatures (B0018 is warmer) and the maximum mostly identifies the battery.
5. **Train baselines:** Fit a training-mean Dummy regressor, Ridge regression and XGBoost on the same split as reference scores.
6. **Train the ANN:** Fully connected Keras network with six hidden dense layers (12-18-24-24-18-12, ReLU, L2 regularization) and a linear capacity output. Features and target are standardized, with Adam, mean squared error and early stopping. A fixed seed makes runs reproducible.
7. **Evaluate:** Randomly split all cycle rows into training, validation and test sets (60/20/20), then compare the ANN with the baselines on MAE (in SOH points) and R², and inspect residuals and SOH predictions per battery.

## 4. Result overview

Results on the 126-row random test split (seed 42):

| Model | MAE [SOH %] | R² |
|---|---|---|
| XGBoost | 0.75 | 0.987 |
| ANN | 1.00 | 0.974 |
| Ridge | 1.28 | 0.972 |
| Training mean (Dummy) | 8.63 | -0.019 |

The test split is random at row level, so neighbouring cycles of the same battery appear in both training and test data. These scores show interpolation between known cycles, not generalization to an unseen battery.

## 5. Conclusions

The features carry a clear aging signal: even Ridge regression reaches R² 0.97, and the nonlinear models improve on it moderately. The ANN does not beat XGBoost, which needs no validation set or early stopping, so the ANN is not justified on this data. The ANN also ran all 100 epochs without early stopping, so the network is probably larger than needed.

Next steps are leave-one-battery-out cross-validation (to measure performance on an unseen battery), ANN runs with several seeds, a smaller network, and testing of RNN or LSTM layers.

## 6. References

[1] G. Kumar, “Estimation of Remaining Useful Life and SOH of Lithium Ion Batteries (For EV Vehicles),” *arXiv* preprint arXiv:2305.10298, May 2023, doi: [10.48550/arXiv.2305.10298](https://doi.org/10.48550/arXiv.2305.10298).

[2] M. F. H. Shiblee and H. Laaksonen, “Battery-Insight-PSO: A machine learning model for accurate prediction of state of health and remaining useful life in lithium-ion batteries,” *Future Batteries*, vol. 8, Art. no. 100114, 2025, doi: [10.1016/j.fub.2025.100114](https://doi.org/10.1016/j.fub.2025.100114).
