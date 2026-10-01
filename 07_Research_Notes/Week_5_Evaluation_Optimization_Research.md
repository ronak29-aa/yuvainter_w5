# Week 5 Research Notes

## Evaluation methodology
The Week 5 workflow uses a chronological holdout and time-series cross-validation rather than random k-fold splitting. This is designed to reduce temporal leakage when predicting future agricultural outcomes.

## Metrics
- MAE: interpretable average absolute prediction error in the target unit.
- RMSE: penalizes larger errors more strongly.
- R²: compares explained variance against a constant-mean baseline.
- MAPE: scale-free percentage error, but should be treated carefully when actual values approach zero.

## Optimization
1. Time-series cross-validation for model-selection stability.
2. GridSearchCV for Random Forest depth, number of trees and minimum leaf size.
3. Ridge regularization over alpha values to control coefficient magnitude in a linear model.
4. Comparison of tuned models against untuned baselines.
5. Segment-level error analysis by crop and state.

## Agricultural interpretation
A model with a lower numerical error is not automatically operationally preferable. Performance should also be checked across crops/regions, against a naïve baseline, for data leakage, and for stability under weather extremes.

## Source research
Government of India OGD crop-production statistics and the Rainfall in India catalog are used as the public-data architecture for the agricultural problem. The Week 4 numeric sample remains synthetic and is not represented as official observations.
