# Benchmark Summary

| Model | MAE | RMSE | Approx. CRPS | 90% Coverage | Forecasts |
|---|---:|---:|---:|---:|---:|
| naive_last | 8.3496 | 12.5206 | 8.2114 | 0.0500 | 40 |
| moving_average_5 | 8.0644 | 12.0983 | 6.9104 | 0.6750 | 40 |
| game_clean_ensemble | 9.1733 | 13.6912 | 9.1157 | 0.0000 | 40 |
| game_attack_identity | 9.3847 | 13.9142 | 9.2763 | 0.0500 | 40 |
| game_attack_ensemble | 9.4802 | 14.0231 | 9.3718 | 0.0250 | 40 |

## Significance Checks

| Comparison | Wilcoxon p-value |
|---|---:|
| game_attack_ensemble vs naive_last | 0.0046 |
| game_attack_ensemble vs game_attack_identity | 0.1576 |
