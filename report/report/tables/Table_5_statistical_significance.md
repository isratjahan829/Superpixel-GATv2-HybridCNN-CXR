> ## NOT REPORTABLE
>
> These numbers come from **synthetic phantom images, not chest X-rays** and **reduced budgets** (epochs=8, cv_epochs=8, bootstrap B=300). They demonstrate that the pipeline executes end to end; they are **not findings** and must not be cited. Re-run on the real dataset at full budget to populate these tables.
>
> Source: `/home/user/Superpixel-GATv2-HybridCNN-CXR/demo_data` (81 images) - device cpu - commit `08d4c8a`

**Table 5. Statistical significance, all pairwise comparisons.**

| Family | Comparison | Metric | Test | Statistic | p-value | Holm-corrected p | Delta (A-B) | Effect size | Significant (alpha=.05) |
|---|---|---|---|---|---|---|---|---|---|
| Split-ratio | S1 vs S2 | accuracy | Paired t-test | t = inf | 0 | 0 | +0.0278 | d = 0.00 (negligible) | Yes |
| Split-ratio | S1 vs S2 | macro_f1 | Paired t-test | t = inf | 0 | 0 | +0.0092 | d = 0.00 (negligible) | Yes |
| Split-ratio | S1 vs S3 | accuracy | Paired t-test | t = 0.00 | 1 | 1 | +0.0000 | d = 0.00 (negligible) | No |
| Split-ratio | S1 vs S4 | accuracy | Paired t-test | t = 0.00 | 1 | 1 | +0.0000 | d = 0.00 (negligible) | No |
| Split-ratio | S1 vs S5 | accuracy | Paired t-test | t = inf | 0 | 0 | +0.0444 | d = 0.00 (negligible) | Yes |
| Split-ratio | S2 vs S4 | accuracy | Paired t-test | t = -inf | 0 | 0 | -0.0278 | d = 0.00 (negligible) | Yes |
| Model-vs-model | GCN vs GraphSAGE | Macro-AUC | DeLong (one-vs-rest, Fisher-combined) | z = -2.44 | 0.0621 | 0.43 | -0.0356 | - | No |
| Model-vs-model | GCN vs GraphSAGE | Accuracy | McNemar (exact) | chi2 = 7.68 | 0.00434 | 0.0434 | +0.1944 | - | Yes (GCN better) |
| Model-vs-model | GCN vs GAT | Macro-AUC | DeLong (one-vs-rest, Fisher-combined) | z = 4.48 | 9.04e-08 | 1.72e-06 | +0.0810 | - | Yes (GCN better) |
| Model-vs-model | GCN vs GAT | Accuracy | McNemar (exact) | chi2 = 18.05 | 1.91e-06 | 3.24e-05 | +0.2778 | - | Yes (GCN better) |
| Model-vs-model | GCN vs GATv2 Hybrid (ours) | Macro-AUC | DeLong (one-vs-rest, Fisher-combined) | z = -1.14 | 0.0649 | 0.43 | +0.0022 | - | No |
| Model-vs-model | GCN vs GATv2 Hybrid (ours) | Accuracy | McNemar (exact) | chi2 = 8.52 | 0.0026 | 0.0286 | +0.2083 | - | Yes (GCN better) |
| Model-vs-model | GCN vs ResNet18 (CNN baseline) | Macro-AUC | DeLong (one-vs-rest, Fisher-combined) | z = 3.84 | 8.63e-05 | 0.00121 | +0.1768 | - | Yes (GCN better) |
| Model-vs-model | GCN vs ResNet18 (CNN baseline) | Accuracy | McNemar (exact) | chi2 = 0.09 | 0.771 | 1 | +0.0417 | - | No |
| Model-vs-model | GraphSAGE vs GAT | Macro-AUC | DeLong (one-vs-rest, Fisher-combined) | z = 5.24 | 2.79e-11 | 5.59e-10 | +0.1167 | - | Yes (GraphSAGE better) |
| Model-vs-model | GraphSAGE vs GAT | Accuracy | McNemar (exact) | chi2 = 4.17 | 0.0312 | 0.281 | +0.0833 | - | No |
| Model-vs-model | GraphSAGE vs GATv2 Hybrid (ours) | Macro-AUC | DeLong (one-vs-rest, Fisher-combined) | z = 1.94 | 0.327 | 0.98 | +0.0379 | - | No |
| Model-vs-model | GraphSAGE vs GATv2 Hybrid (ours) | Accuracy | McNemar (exact) | chi2 = 0.00 | 1 | 1 | +0.0139 | - | No |
| Model-vs-model | GraphSAGE vs ResNet18 (CNN baseline) | Macro-AUC | DeLong (one-vs-rest, Fisher-combined) | z = 4.78 | 1.58e-05 | 0.000237 | +0.2124 | - | Yes (GraphSAGE better) |
| Model-vs-model | GraphSAGE vs ResNet18 (CNN baseline) | Accuracy | McNemar (exact) | chi2 = 3.45 | 0.0614 | 0.43 | -0.1528 | - | No |
| Model-vs-model | GAT vs GATv2 Hybrid (ours) | Macro-AUC | DeLong (one-vs-rest, Fisher-combined) | z = -3.15 | 3.24e-06 | 5.19e-05 | -0.0788 | - | Yes (GATv2 Hybrid (ours) better) |
| Model-vs-model | GAT vs GATv2 Hybrid (ours) | Accuracy | McNemar (exact) | chi2 = 2.29 | 0.125 | 0.5 | -0.0694 | - | No |
| Model-vs-model | GAT vs ResNet18 (CNN baseline) | Macro-AUC | DeLong (one-vs-rest, Fisher-combined) | z = 2.16 | 5.73e-07 | 1.03e-05 | +0.0958 | - | Yes (GAT better) |
| Model-vs-model | GAT vs ResNet18 (CNN baseline) | Accuracy | McNemar (exact) | chi2 = 9.48 | 0.00151 | 0.0182 | -0.2361 | - | Yes (ResNet18 (CNN baseline) better) |
| Model-vs-model | GATv2 Hybrid (ours) vs ResNet18 (CNN baseline) | Macro-AUC | DeLong (one-vs-rest, Fisher-combined) | z = 3.48 | 0.000768 | 0.00999 | +0.1746 | - | Yes (GATv2 Hybrid (ours) better) |
| Model-vs-model | GATv2 Hybrid (ours) vs ResNet18 (CNN baseline) | Accuracy | McNemar (exact) | chi2 = 4.32 | 0.0357 | 0.286 | -0.1667 | - | No |

Holm-Bonferroni correction is applied within each family. The DeLong row reports the Stouffer-combined z across one-vs-rest classes (signed, so the direction is readable) while the p-value is Fisher-combined; McNemar is the exact paired test on the same held-out instances.
