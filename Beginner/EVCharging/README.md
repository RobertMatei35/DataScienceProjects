# Residential EV Charging Patterns

## 1. Project description

This notebook explores residential electric-vehicle charging behavior using unsupervised learning. Its main goals are to clean and understand charging datasets, engineer session-level features, cluster charging sessions into interpretable patterns, and investigate anomalous sessions. It also compares charging behavior for private and shared charging ports.

## 2. Data description

The datasets are described in the article [Residential electric vehicle charging datasets from apartment buildings](https://www.sciencedirect.com/science/article/pii/S2352340921003899). The notebook loads the supplied CSV files from `datasets/archive`.

The core charging reports cover 6,878 sessions from 97 users across 24 garages in Trondheim, Norway, from December 2018 to January 2020. Session fields include user and garage identifiers, private/shared charging-port type, plug-in and plug-out timestamps, and energy charged (`El_kWh`).

Additional files provide hourly charging loads and idle capacity, smart-meter electricity use, nearby traffic counts, and local weather observations. The notebook inspects and prepares these sources, while its final session-clustering features are derived from the charging-session records.

## 3. Project motivation and objective

The dataset is documented in [1], which describes its structure and available features. The study in [2] demonstrates how charging reports and smart-meter data can be used to examine charging habits and energy flexibility; it inspired this project.

This notebook explores whether charging sessions form distinct patterns without using labeled examples. The goal is exploratory: to identify and interpret patterns in this dataset, not to predict predefined categories or claim that the clusters apply to all EV users.

## 4. Machine-learning workflow

1. **Load and inspect the data:** Read the charging, aggregated load, smart-meter, traffic, and weather files; review data types and missing values.
2. **Prepare the datasets:** Convert timestamps and numeric columns, remove sessions with missing plug-out times, fill missing shared-port IDs for private sessions, and handle missing hourly measurements according to their meaning in the source data.
3. **Explore and engineer session features:** Calculate session duration and average charging power, encode plug-in and plug-out times cyclically, mark weekends, and apply log transforms to skewed duration, energy, and power features. Flag unusually long sessions and remove very short, low-energy records from clustering.
4. **Scale and cluster:** Standardize the session features and reduce the weekend feature's weight. Compare K-Means cluster counts, then evaluate HDBSCAN and Gaussian Mixture Models as alternatives.
5. **Inspect anomalies and charging types:** Apply Isolation Forest to identify unusual sessions, inspect the flagged records, and compare descriptive statistics for private and shared charging sessions.

## 5. Result overview

- **K-Means:** Three clusters were selected as a useful and interpretable solution. For k=3, the notebook reports a silhouette score of **0.317** and a Davies–Bouldin score of **1.329**. The repeated-seed stability score is **0.999**.
- **Session patterns:** The clusters are interpreted as midday charging (about 12% of sessions), after-work charging (31%), and overnight charging (57%). The overnight group has lower average charging power, consistent with longer plug-in periods.
- **Alternative clustering:** HDBSCAN and Gaussian Mixture Model results were less clearly separated or less interpretable than the K-Means solution in this analysis.
- **Anomaly detection:** Isolation Forest flags **35.35%** of sessions with its automatic contamination setting. The notebook considers this too many to treat all flagged records as genuine anomalies without further investigation.
- **Private vs. shared ports:** The notebook reports longer average sessions for private ports than shared ports (about 12.78 vs. 6.53 hours), while estimated average power is lower (about 0.87 vs. 2.85 kW).

## 6. Conclusions

The notebook demonstrates how unsupervised methods can help summarize charging behavior into time-of-day and duration patterns. K-Means produced the clearest three-cluster interpretation in this experiment, while the anomaly-detection results show why algorithmic outlier labels need domain review rather than automatic acceptance.

These findings describe a limited dataset from apartment garages in one city and period, so they should not be assumed to represent all EV drivers or charging infrastructure. Useful next steps include clustering users as well as sessions, combining user- and session-level behavior, checking the influence of private versus shared charging, and validating anomaly thresholds with domain knowledge.

## 7. References

[1] Å. L. Sørensen, K. B. Lindberg, I. Sartori, and I. Andresen, “Residential electric vehicle charging datasets from apartment buildings,” *Data in Brief*, vol. 36, Art. no. 107105, 2021, doi: [10.1016/j.dib.2021.107105](https://doi.org/10.1016/j.dib.2021.107105).

[2] Å. L. Sørensen, K. B. Lindberg, I. Sartori, and I. Andresen, “Analysis of residential EV energy flexibility potential based on real-world charging reports and smart meter data,” *Energy and Buildings*, vol. 241, Art. no. 110923, 2021, doi: [10.1016/j.enbuild.2021.110923](https://doi.org/10.1016/j.enbuild.2021.110923).
