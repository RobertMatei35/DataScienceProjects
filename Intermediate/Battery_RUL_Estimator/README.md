# Battery Remaining Useful Life (RUL) Estimator

## 1. Project description

This notebook explores using a Keras artificial neural network (ANN) to estimate battery aging from charge and discharge measurements. It builds cycle-level features for battery B0005 and trains the network to predict discharge capacity, from which State of Health (SOH) and Remaining Useful Life (RUL) are derived. The work is inspired by Kumar's study of battery SOH and RUL estimation [1]. For additional context on machine-learning approaches to these tasks, see [2].

The notebook focuses on one battery and is an exploratory modeling project, not an industry-validated RUL system. The model predicts capacity; it does not directly train on RUL as its target.

## 2. Data description

The notebook downloads the [NASA Battery dataset on Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset), which provides battery cycling data in CSV form. The referenced [NASA Battery Aging dataset files](https://labinfo.ing.he-arc.ch/gitlab/ticc/16TICc19/nasa-battery-dataset/-/tree/614f3518915079a5182655cadc3492973ac90d6c/data/BatteryAgingARC-FY08Q4) contain charge, discharge, and impedance measurements across repeated cycles. This notebook focuses on battery **B0005**.

Measurements include voltage, current, temperature, and time during charge and discharge operations. Discharge capacity is used as the model target. SOH is calculated relative to an initial capacity of 2 Ah, and the notebook creates an RUL value from the cycle's position in the battery's recorded lifetime.

## 3. Machine-learning workflow

1. **Load and inspect the data:** Download the CSV dataset, use its metadata to identify B0005 cycle files, and examine sample charge and discharge profiles.
2. **Prepare cycle pairs:** Inspect and remove duplicate operations, then pair consecutive charge and discharge records to represent complete cycles.
3. **Engineer features:** Extract charge constant-current and constant-voltage durations, charge-start voltage, maximum charge and discharge temperatures, minimum discharge voltage, and the rest interval between cycles. Derive SOH and RUL from capacity and cycle order.
4. **Select model inputs:** Use seven standardized inputs—charge CC and CV times, starting voltage, maximum charge and discharge temperatures, minimum discharge voltage, and a transformed rest-time feature—to predict capacity. Discharge time is excluded because, under the fixed-current conditions in the data, it is directly related to capacity.
5. **Train the ANN:** Train a fully connected Keras network with five hidden dense layers and a linear capacity output. Standardize features and target, use Adam with mean squared error, and apply early stopping.
6. **Evaluate:** Randomly split cycle rows into training and test subsets (80/20), then compare predicted and measured capacity and inspect residuals and derived SOH.

## 4. Result overview

The notebook reports an **R² of 0.9845** for capacity predictions on its 34-row test split. This is a result from one random split of B0005 cycle records, not evidence that the model generalizes to other batteries.

The test subset is also passed to Keras as validation data for early stopping, so it is used during model training as well as for the reported evaluation. The result should therefore be treated as an exploratory score rather than a fully independent test estimate.

## 5. Conclusions

The notebook demonstrates a complete feature-engineering and ANN workflow for battery-cycle data, with capacity as an intermediate measure for SOH and RUL. Its high saved R² is encouraging for this single-battery experiment, but the small, randomly split set of cycles and reuse of the test subset for early stopping limit what can be concluded.

Next steps consists of the inclusion of multiple datasets B006, B007, B0018 that were tested in the same condition as B0005 and testing of RNN or LSTM neurons. The usage of increased number of datasets and various types of neurons will demonstrate the viability of the ANN. 

## 6. References

[1] G. Kumar, “Estimation of Remaining Useful Life and SOH of Lithium Ion Batteries (For EV Vehicles),” *arXiv* preprint arXiv:2305.10298, May 2023, doi: [10.48550/arXiv.2305.10298](https://doi.org/10.48550/arXiv.2305.10298).

[2] M. F. H. Shiblee and H. Laaksonen, “Battery-Insight-PSO: A machine learning model for accurate prediction of state of health and remaining useful life in lithium-ion batteries,” *Future Batteries*, vol. 8, Art. no. 100114, 2025, doi: [10.1016/j.fub.2025.100114](https://doi.org/10.1016/j.fub.2025.100114).
