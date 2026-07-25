> ## NOT REPORTABLE
>
> These numbers come from **synthetic phantom images, not chest X-rays** and **reduced budgets** (epochs=8, cv_epochs=8, bootstrap B=300). They demonstrate that the pipeline executes end to end; they are **not findings** and must not be cited. Re-run on the real dataset at full budget to populate these tables.
>
> Source: `/home/user/Superpixel-GATv2-HybridCNN-CXR/demo_data` (81 images) - device cpu - commit `08d4c8a`

**Table 4. Confusion matrix, best split configuration (S1), pooled over seeds (rows = true, columns = predicted).**

| True \ Predicted | Cardiac Pathology | Chronic Lung Disease | Normal | Pleural Pathology | Tuberculosis (TB) | Total | Recall |
|---|---|---|---|---|---|---|---|
| Cardiac Pathology | 0 | 0 | 0 | 4 | 0 | 4 | 0.000 |
| Chronic Lung Disease | 0 | 0 | 0 | 2 | 0 | 2 | 0.000 |
| Normal | 0 | 0 | 0 | 6 | 0 | 6 | 0.000 |
| Pleural Pathology | 0 | 0 | 0 | 2 | 0 | 2 | 1.000 |
| Tuberculosis (TB) | 0 | 0 | 0 | 4 | 0 | 4 | 0.000 |
| **Total predicted** | 0 | 0 | 0 | 18 | 0 | 18 | - |
| **Precision** | n/a | n/a | n/a | 0.111 | n/a | - | 0.111 (accuracy) |
