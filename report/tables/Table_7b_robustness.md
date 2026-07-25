> ## NOT REPORTABLE
>
> These numbers come from **synthetic phantom images, not chest X-rays** and **reduced budgets** (epochs=8, cv_epochs=8, bootstrap B=300). They demonstrate that the pipeline executes end to end; they are **not findings** and must not be cited. Re-run on the real dataset at full budget to populate these tables.
>
> Source: `/home/user/Superpixel-GATv2-HybridCNN-CXR/demo_data` (81 images) - device cpu - commit `08d4c8a`

**Table 7b. Robustness: performance under input perturbation.**

| Perturbation | Accuracy | Delta Acc. | Macro F1 | Macro AUC | MCC | Retention | Grade |
|---|---|---|---|---|---|---|---|
| None (clean test set) | 0.4444 | - | 0.4889 | 0.9429 | 0.4637 | 100.0% | Reference |
| Gaussian noise (sigma=0.02) | 0.4444 | +0.0000 | 0.4889 | 0.9429 | 0.4637 | 100.0% | A |
| Gaussian noise (sigma=0.05) | 0.2222 | -0.2222 | 0.2571 | 1.0000 | 0.1406 | 50.0% | D |
| JPEG compression (q=50) | 0.4444 | +0.0000 | 0.4889 | 0.9286 | 0.4637 | 100.0% | A |
| Rotation (+/-10 deg) | 0.4444 | +0.0000 | 0.5000 | 0.8952 | 0.3723 | 100.0% | A |
| Brightness shift (+/-20%) | 0.1111 | -0.3333 | 0.0400 | 0.7663 | 0.0000 | 25.0% | D |
| Contrast reduction (-30%) | 0.6667 | +0.2222 | 0.5100 | 0.9429 | 0.5992 | 150.0% | A |

Retention = perturbed accuracy / clean accuracy, on 9 held-out test images. Grades: A >= 96%, B >= 92%, C >= 88%, D below. Perturbations are applied to the image and the graph is rebuilt through the identical SLIC + encoder path.
