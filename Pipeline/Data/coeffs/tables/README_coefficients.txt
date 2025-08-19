Aggregated coefficient artefacts
--------------------------------
Root: D:\Python - Project\Pipeline
Scenarios: raw, scaled, zscore

Generated files:
- coefficients_long.csv: one row per (scenario, model_tag, feature) with raw coefficients.
- coefficients_wide_by_model.csv: features as rows; columns are (scenario, model_tag) pairs.
- coefficients_consensus_by_model.csv: mean/std of each feature's coefficient across scenarios, per model.
- coefficients_top_abs_mean.csv: Top-N features per model by absolute consensus coefficient mean.

Notes:
- Only M1/M2 produce coefficients.csv (GAM/RF use shapes/importances instead).
- Intercepts are not included because the exporter writes only feature-level coefficients.
