# Causal-ablation summary table

Pre-registered thresholds:
- AUROC > 0.95 = weak dependence
- AUROC 0.85–0.95 = moderate
- AUROC < 0.85 = strong dependence
- |Δ| reduction > 50% = strong; 20–50% = moderate; <20% = weak.

## AUROC × condition × ablation (per session)

### D2

| ablation | fake_5k | fake_25k | fake_70k | fake_100k | shuffled_E | cross_session_E |
|---|---|---|---|---|---|---|
| E1 | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak |
| E2 | 0.993 _moderate_ | 0.990 _moderate_ | 0.987 _moderate_ | 0.973 _moderate_ | 0.996 weak | 0.997 weak |
| E3 | 0.881 **strong** | 0.802 **strong** | 0.822 **strong** | 0.789 **strong** | 0.896 **strong** | 0.896 **strong** |
| E4 | 1.000 **strong** | 1.000 **strong** | 0.931 **strong** | 0.151 **strong** | 0.478 **strong** | 0.477 **strong** |
| E5 | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak |

### V10

| ablation | fake_5k | fake_25k | fake_70k | fake_100k | shuffled_E | cross_session_E |
|---|---|---|---|---|---|---|
| E1 | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak |
| E2 | 0.983 _moderate_ | 0.983 _moderate_ | 0.983 _moderate_ | 0.983 _moderate_ | 0.995 weak | 0.995 weak |
| E3 | 0.977 **strong** | 0.930 **strong** | 0.948 **strong** | 0.926 **strong** | 0.943 **strong** | 0.938 **strong** |
| E4 | 1.000 _moderate_ | 1.000 **strong** | 1.000 **strong** | 0.904 **strong** | 0.536 **strong** | 0.549 **strong** |
| E5 | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak | 1.000 weak |
