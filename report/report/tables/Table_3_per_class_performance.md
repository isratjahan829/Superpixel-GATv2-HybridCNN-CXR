> ## NOT REPORTABLE
>
> These numbers come from **synthetic phantom images, not chest X-rays** and **reduced budgets** (epochs=8, cv_epochs=8, bootstrap B=300). They demonstrate that the pipeline executes end to end; they are **not findings** and must not be cited. Re-run on the real dataset at full budget to populate these tables.
>
> Source: `/home/user/Superpixel-GATv2-HybridCNN-CXR/demo_data` (81 images) - device cpu - commit `08d4c8a`

**Table 3. Per-class performance, best split configuration (S1), pooled over 2 seeds. Wilson score 95% CIs.**

| Class | Support | Prevalence | Sensitivity (95% CI) | Specificity (95% CI) | PPV (95% CI) | NPV (95% CI) | F1 | AUC | LR+ | LR- |
|---|---|---|---|---|---|---|---|---|---|---|
| Cardiac Pathology | 4 | 22.2% | 0.000 (0.000-0.490) | 1.000 (0.785-1.000) | n/a (n/a-n/a) | 0.778 (0.548-0.910) | 0.000 | 0.411 | inf | 1.00 |
| Chronic Lung Disease | 2 | 11.1% | 0.000 (0.000-0.658) | 1.000 (0.806-1.000) | n/a (n/a-n/a) | 0.889 (0.672-0.969) | 0.000 | 0.750 | inf | 1.00 |
| Normal | 6 | 33.3% | 0.000 (0.000-0.390) | 1.000 (0.758-1.000) | n/a (n/a-n/a) | 0.667 (0.437-0.837) | 0.000 | 0.819 | inf | 1.00 |
| Pleural Pathology | 2 | 11.1% | 1.000 (0.342-1.000) | 0.000 (0.000-0.194) | 0.111 (0.031-0.328) | n/a (n/a-n/a) | 0.200 | 1.000 | 1.00 | n/a |
| Tuberculosis (TB) | 4 | 22.2% | 0.000 (0.000-0.490) | 1.000 (0.785-1.000) | n/a (n/a-n/a) | 0.778 (0.548-0.910) | 0.000 | 0.696 | inf | 1.00 |
| **Macro average** | 18 | 100% | 0.200 | 0.800 | 0.111 | 0.778 | 0.040 | 0.735 | 1.00 | 1.00 |

Wilson score intervals are used for every proportion: more reliable than the normal approximation at the sample sizes of the minority classes. Sensitivity here is the same quantity as per-class recall in Table 4 and is computed from the same pooled predictions, so the two tables agree by construction.
