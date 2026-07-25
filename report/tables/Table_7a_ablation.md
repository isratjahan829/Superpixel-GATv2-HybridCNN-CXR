> ## NOT REPORTABLE
>
> These numbers come from **synthetic phantom images, not chest X-rays** and **reduced budgets** (epochs=8, cv_epochs=8, bootstrap B=300). They demonstrate that the pipeline executes end to end; they are **not findings** and must not be cited. Re-run on the real dataset at full budget to populate these tables.
>
> Source: `/home/user/Superpixel-GATv2-HybridCNN-CXR/demo_data` (81 images) - device cpu - commit `08d4c8a`

**Table 7a. Ablation study: contribution of each component.**

| Configuration | Accuracy | Macro F1 | Macro AUC | MCC | Delta Acc. vs reference | Trainable params | Infer. (ms) |
|---|---|---|---|---|---|---|---|
| Full GATv2 Hybrid (reference) | 0.1533 | 0.1931 | 0.8202 | 0.1989 | - | 79,293 | 2.5 |
| - GATv2 -> GCN layer | 0.3619 | 0.3278 | 0.8355 | 0.3166 | +0.2086 | 41,997 | 1.7 |
| - Multi-head attention (1 head) | 0.1248 | 0.1031 | 0.6977 | 0.0923 | -0.0286 | 23,997 | 2.2 |
| - Class weighting / sampler | 0.3457 | 0.1550 | 0.5424 | 0.1713 | +0.1924 | 79,293 | 2.8 |
| - Superpixel graph (CNN only) | 0.3181 | 0.1650 | 0.7481 | 0.1598 | +0.1648 | 11,179,077 | n/a |
| Hand-only (12d) +edge | 0.2781 | 0.1216 | 0.6873 | 0.0698 | +0.1248 | 29,885 | 2.4 |
| Deep-only (128d) +edge | 0.1676 | 0.1600 | 0.8202 | 0.1816 | +0.0143 | 74,661 | 2.5 |
| Hybrid (140d) +edge | 0.1533 | 0.1931 | 0.8202 | 0.1989 | +0.0000 | 79,293 | 2.7 |
| Hybrid (140d) no-edge | 0.1676 | 0.2335 | 0.7985 | 0.2447 | +0.0143 | 78,813 | 2.2 |

Each arm removes exactly one component and is trained on the same folds with the same budget, so the delta is attributable to that component. A positive delta means the removed component was not helping.
