# LearnTrack AI: Predictive Academic Modeling

LearnTrack AI is an advanced machine learning pipeline designed to predict student pass probabilities based on their academic behaviors and lab report submission patterns. 

This project utilizes a powerful ensemble of gradient boosting frameworks (LightGBM, XGBoost, CatBoost) to maximize the Area Under the Precision-Recall Curve (AUPRC).

## Pipeline Workflow

- **Data Loading**: Engineered over 50 behavioral and time-series features directly from raw log data. This includes temporal patterns (time of day, weekend vs. weekday submissions), grade trajectories, performance momentum, rolling statistics, and efficiency scores.
- **Feature Engineering**: Implements a Stratified 7-Fold Cross Validation approach to ensure stable and generalizable performance across different student cohorts.
- **Model Training**:
  - Folds the data into 7 stratified splits.
  - Trains LightGBM, XGBoost, and CatBoost model concurrently.
  - Tracks out-of-fold (OOF) predictions to compute the OOF AUPRC.
