> ## NOT REPORTABLE
>
> These numbers come from **synthetic phantom images, not chest X-rays** and **reduced budgets** (epochs=8, cv_epochs=8, bootstrap B=300). They demonstrate that the pipeline executes end to end; they are **not findings** and must not be cited. Re-run on the real dataset at full budget to populate these tables.
>
> Source: `/home/user/Superpixel-GATv2-HybridCNN-CXR/demo_data` (81 images) - device cpu - commit `08d4c8a`

**Table 2. Stratified 5-fold cross-validation.**

| Fold | Train n | Val n | Test n | Accuracy | Balanced Acc. | Macro F1 | Macro AUC | MCC |
|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 9 | 15 | 0.1333 | 0.4000 | 0.2267 | 0.8149 | 0.2200 |
| 2 | 48 | 9 | 15 | 0.1333 | 0.4000 | 0.2267 | 0.7713 | 0.2200 |
| 3 | 49 | 9 | 14 | 0.1429 | 0.4000 | 0.2286 | 0.8533 | 0.2288 |
| 4 | 49 | 9 | 14 | 0.2857 | 0.4000 | 0.2571 | 0.7795 | 0.3257 |
| 5 | 49 | 9 | 14 | 0.0714 | 0.2000 | 0.0267 | 0.8818 | 0.0000 |
| **Mean +/- SD** | - | - | - | 0.1533 +/- 0.0709 | 0.3600 +/- 0.0800 | 0.1931 +/- 0.0840 | 0.8202 +/- 0.0424 | 0.1989 +/- 0.1072 |

Folds are stratified on the original (non-augmented) images only; augmented graphs never enter a held-out fold.
