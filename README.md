# Week 5 — Model Evaluation, Optimization and Reporting in Agribusiness

## Main task
Evaluate and optimize the Week 4 crop-yield prediction model.

## Workflow
1. Chronological holdout: train through 2022; test on 2023–2025.
2. Compare baseline/candidate models.
3. Run five-fold time-series cross-validation on the training period.
4. Tune Random Forest hyperparameters.
5. Tune Ridge regularization.
6. Analyze residuals by crop, state and year.
7. Compare optimized models.
8. Produce an executive-style report.

## Selected optimized candidate in this run
Tuned Ridge

## Important data note
The included agricultural numerical data are synthetic demonstration data derived from the Week 4 workflow. They are not presented as official observations. Public-data sources and methodology are documented separately.

## Run
From `04_Code`:
`python week5_evaluation_optimization.py`

Required packages:
- pandas
- numpy
- scikit-learn
