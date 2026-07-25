> ## NOT REPORTABLE
>
> These numbers come from **synthetic phantom images, not chest X-rays** and **reduced budgets** (epochs=8, cv_epochs=8, bootstrap B=300). They demonstrate that the pipeline executes end to end; they are **not findings** and must not be cited. Re-run on the real dataset at full budget to populate these tables.
>
> Source: `/home/user/Superpixel-GATv2-HybridCNN-CXR/demo_data` (81 images) - device cpu - commit `08d4c8a`

**Table 7c. Error analysis: misclassification taxonomy.**

| Error pattern | Count | % of errors | Mean confidence |
|---|---|---|---|
| Normal -> Pleural Pathology | 6 | 37.5% | 0.274 |
| Cardiac Pathology -> Pleural Pathology | 4 | 25.0% | 0.300 |
| Tuberculosis (TB) -> Pleural Pathology | 4 | 25.0% | 0.273 |
| Chronic Lung Disease -> Pleural Pathology | 2 | 12.5% | 0.265 |
| **Total errors** | 16 | 100% | 0.279 |

Errors are 88.9% of 18 pooled predictions. Mean confidence on correct predictions is 0.337 vs 0.279 on errors. Abstaining below confidence 0.55 would defer 100.0% of errors, at the cost of deferring 100.0% of correct predictions.
