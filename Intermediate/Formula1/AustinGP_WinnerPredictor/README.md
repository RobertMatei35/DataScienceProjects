# Austin Grand Prix Winner Predictor

## 1. Project description

This project explores whether historical Formula 1 results can help rank drivers by their relative strength to win the United States Grand Prix at Austin. It starts from the foundation laid by Mariana Antaya in her webinar; her original [Google Colab notebook](https://colab.research.google.com/drive/1LCRUTXCdFeNUbsTaUxZobRS_xXJJ9-xf?usp=sharing) is linked here.

This project includes both the notebook from the webinar, kept under its original name [Austin_GP_Winner_Predictor](./Austin_GP_Winner_Predictor.ipynb), and my version, [Austin_GP_Winner_Predictor_MyTry](./Austin_GP_Winner_Predictor_MyTry.ipynb).

The notebook extends that starting point with API-based data collection, features built from driver, team, and Austin circuit history, a chronological holdout evaluation, and a comparison with a simple recent-form baseline. Its scores are rankings, not calibrated win probabilities or predictions of finishing positions.

## 2. Data description

Race schedules and driver results are retrieved through the Ergast API using FastF1's Ergast interface. The notebook loads results from the configured start season through the target season. Each row represents one driver's result in one race, including the driver and team, finishing position, points, race, and date. Sprint results are excluded.

The notebook checks that race results are complete and uses only races before the target Austin event. It derives its modeling features from past finishes, points, wins, podiums, team results, and driver history at the Austin circuit.

## 3. Machine-learning workflow

1. **Configure the prediction:** Set the target year and history start year; reserve the previous full season for testing.
2. **Collect and validate results:** Fetch the target season schedule and race results, identify the United States Grand Prix at Austin, exclude the target race and later results, and check that each included race has complete entries and one winner.
3. **Build historical features:** Calculate nine pre-race features, including each driver's recent form over up to five races, recent team form, the driver's Austin history over up to three visits, and the driver's previous-race count. Features are built before adding each race's outcomes to history to avoid using future results.
4. **Train and test:** Train an XGBoost binary classifier on older seasons and evaluate it on the preceding full season. For each test race, select the driver with the highest model score and compare this choice with a baseline that selects the driver with the best recent average finish.
5. **Rank target-year drivers:** Train on all eligible historical results before Austin, score the available driver entries, and rank the field. The notebook defaults to entries from the most recent completed race; users can supply an updated `austin_entries.csv` when needed.

## 4. Result overview

On the notebook's saved holdout evaluation for the **2025 season (24 races)**:

| Method | Race winners correctly selected |
|---|---:|
| XGBoost classifier | 6 / 24 (25.0%) |
| Best recent average-finish baseline | 7 / 24 (29.2%) |

The simple recent-form baseline selected more winners than the classifier in this holdout. The notebook also generates a top-three ranking for the configured Austin race, but these ranks are model scores—not confirmed win probabilities or predicted podium positions. Refresh the data and entry list before using a run for a future race.

## 5. Conclusions

This project demonstrates an end-to-end motorsport machine-learning workflow: collecting results from an API, creating pre-race features, evaluating chronologically, and comparing a model with a simple baseline. In the saved test, the classifier did not outperform that baseline, so its rankings should be treated as exploratory rather than reliable forecasts.

Future work could test more seasons with walk-forward evaluation, improve feature design and calibration, compare additional models, and rethink the target variable from categorical driver name to winning probability
