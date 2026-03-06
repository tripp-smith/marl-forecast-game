# Benchmark Summary

| Model | MAE | RMSE | Approx. CRPS | 90% Coverage | Forecasts |
|---|---:|---:|---:|---:|---:|
| naive_last | 3.1749 | 3.8071 | 3.0386 | 0.0500 | 40 |
| moving_average_5 | 2.8331 | 3.4051 | 1.9934 | 0.9250 | 40 |
| game_clean_ensemble | 2.9722 | 3.4965 | 2.9150 | 0.0250 | 40 |
| game_attack_identity | 2.9017 | 3.4725 | 2.8055 | 0.1000 | 40 |
| game_attack_ensemble | 2.9327 | 3.5027 | 2.8343 | 0.0750 | 40 |

## Significance Checks

| Comparison | Wilcoxon p-value |
|---|---:|
| game_attack_ensemble vs naive_last | 0.6380 |
| game_attack_ensemble vs game_attack_identity | 0.5537 |
