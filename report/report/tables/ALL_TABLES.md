# Evaluation tables

> ## NOT REPORTABLE
>
> These numbers come from **synthetic phantom images, not chest X-rays** and **reduced budgets** (epochs=8, cv_epochs=8, bootstrap B=300). They demonstrate that the pipeline executes end to end; they are **not findings** and must not be cited. Re-run on the real dataset at full budget to populate these tables.
>
> Source: `/home/user/Superpixel-GATv2-HybridCNN-CXR/demo_data` (81 images) - device cpu - commit `08d4c8a`

Every number below is read from an artifact in `/home/user/Superpixel-GATv2-HybridCNN-CXR/outputs`; none is transcribed by hand.

**Table 1. Split configurations, aggregate performance (mean +/- SD over 2 seeds) and bootstrapped 95% CIs.**

| Config | Ratio (Tr/Va/Te) | n (Tr/Va/Te) | Accuracy | Bal. Acc. | Macro F1 | Macro AUC | Accuracy 95% CI | Macro F1 95% CI | Macro AUC 95% CI | MCC 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 80%/10%/10% | 63 / 9 / 9 | 0.1111 +/- 0.0000 | 0.2000 +/- 0.0000 | 0.0400 +/- 0.0000 | 0.7683 +/- 0.0063 | 0.111 (0.000-0.278) | 0.040 (0.000-0.093) | 0.735 (0.548-0.855) | 0.000 (0.000-0.000) |
| S2 | 70%/15%/15% | 57 / 12 / 12 | 0.0833 +/- 0.0000 | 0.2000 +/- 0.0000 | 0.0308 +/- 0.0000 | 0.8477 +/- 0.0181 | 0.083 (0.000-0.208) | 0.031 (0.000-0.070) | 0.759 (0.610-0.871) | 0.000 (0.000-0.000) |
| S3 | 75%/15%/10% | 60 / 12 / 9 | 0.1111 +/- 0.0000 | 0.2000 +/- 0.0000 | 0.0400 +/- 0.0000 | 0.7825 +/- 0.0254 | 0.111 (0.000-0.278) | 0.040 (0.000-0.093) | 0.724 (0.535-0.853) | 0.000 (0.000-0.000) |
| S4 | 70%/20%/10% | 57 / 15 / 9 | 0.1111 +/- 0.0000 | 0.2000 +/- 0.0000 | 0.0400 +/- 0.0000 | 0.8413 +/- 0.0016 | 0.111 (0.000-0.278) | 0.040 (0.000-0.093) | 0.756 (0.561-0.888) | 0.000 (0.000-0.000) |
| S5 | 60%/20%/20% | 51 / 15 / 15 | 0.0667 +/- 0.0000 | 0.2000 +/- 0.0000 | 0.0250 +/- 0.0000 | 0.8313 +/- 0.0542 | 0.067 (0.000-0.167) | 0.025 (0.000-0.059) | 0.693 (0.548-0.825) | 0.000 (0.000-0.000) |
| **Mean** | - | 81 total | 0.0967 +/- 0.0185 | 0.2000 +/- 0.0000 | 0.0352 +/- 0.0062 | 0.8142 +/- 0.0324 | - | - | - | - |

All splits are stratified by class and performed on raw images before augmentation (no image-level leakage). Seeds 42-43. Best configuration by mean accuracy: **S1**; bootstrap B = 300.

---

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

---

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

---

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

---

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

---

**Table 6. Model comparison: discrimination, agreement, calibration and cost.**

| Model | Acc. | Macro F1 | Macro AUC | Bal. Acc. | MCC (95% CI) | Cohen's kappa (95% CI) | Youden J | Brier | ECE | Log loss | Trainable params | Train (min/fold) | Infer. (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GCN | 0.361 | 0.388 | 0.816 | 0.467 | 0.333 (0.221-0.433) | 0.169 (0.074-0.281) | 0.296 | 0.768 | 0.103 | 1.533 | 41,997 | 0.01 | 1.7 |
| GraphSAGE | 0.167 | 0.195 | 0.852 | 0.400 | 0.206 (0.094-0.299) | 0.091 (0.031-0.158) | 0.218 | 0.810 | 0.198 | 1.649 | 78,093 | 0.01 | 1.3 |
| GAT | 0.083 | 0.031 | 0.735 | 0.200 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.000 | 0.824 | 0.256 | 1.671 | 43,197 | 0.01 | 2.2 |
| GATv2 Hybrid (ours) | 0.153 | 0.215 | 0.814 | 0.367 | 0.223 (0.106-0.295) | 0.076 (0.016-0.135) | 0.182 | 0.827 | 0.233 | 1.686 | 79,293 | 0.01 | 2.4 |
| ResNet18 (CNN baseline) | 0.319 | 0.185 | 0.639 | 0.267 | 0.094 (-0.015-0.214) | 0.071 (-0.011-0.161) | 0.081 | 0.790 | 0.094 | 1.585 | 11,179,077 | 0.05 | 10.3 |

MCC and Cohen's kappa stay meaningful under class imbalance, unlike raw accuracy. Brier score, ECE and log loss quantify how trustworthy the probabilities are, which any claim of clinical usefulness depends on. Parameter counts are TRAINABLE parameters: the GATv2 hybrid's ResNet18 encoder is frozen and therefore excluded, which is why its count is far below the fine-tuned CNN baseline's. Timings are for the hardware this run used and are not comparable across machines.

---

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

---

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

---

**Table 7c. Error analysis: misclassification taxonomy.**

| Error pattern | Count | % of errors | Mean confidence |
|---|---|---|---|
| Normal -> Pleural Pathology | 6 | 37.5% | 0.274 |
| Cardiac Pathology -> Pleural Pathology | 4 | 25.0% | 0.300 |
| Tuberculosis (TB) -> Pleural Pathology | 4 | 25.0% | 0.273 |
| Chronic Lung Disease -> Pleural Pathology | 2 | 12.5% | 0.265 |
| **Total errors** | 16 | 100% | 0.279 |

Errors are 88.9% of 18 pooled predictions. Mean confidence on correct predictions is 0.337 vs 0.279 on errors. Abstaining below confidence 0.55 would defer 100.0% of errors, at the cost of deferring 100.0% of correct predictions.

---

**Table 8. Literature benchmarking (qualitative positioning).**

| Study / Model | Year | Dataset (n) | Task / Classes | Approach | Acc. | Macro AUC | Macro F1 | Sens. (macro) |
|---|---|---|---|---|---|---|---|---|
| CheXNet (Rajpurkar et al., 2017) | 2017 | ChestX-ray14 (112,120 images) | 14 findings, multi-label binary | 121-layer DenseNet | n/a (multi-label) | ~0.83-0.84 mean AUROC | 0.435 (pneumonia) | n/a |
| COVID-Net (Wang, Lin & Wong, 2020) | 2020 | COVIDx (~13,975 images) | 3-class multiclass | Lightweight CNN (PEPX) | 0.933 | n/a | n/a | ~0.91 (COVID class) |
| MIMIC-CXR single-source benchmark | recent | MIMIC-CXR (~377,110 images) | Multi-label findings | CNN multi-label classifier | n/a | ~0.75 mean AUROC | n/a | n/a |
| **GATv2 Hybrid (this work)** | this study | This dataset (81) | 5-class multiclass | Frozen ResNet18 features + superpixel GATv2 | 0.444 | 0.943 | 0.489 | 0.600 |

*Because label sets, task framing and dataset scale differ by two to three orders of magnitude, this table supports only relative discussion (direction and size of the gap, with the dataset-scale caveat), never a claim of superiority. The only controlled head-to-head comparison in this work is the CNN baseline in Table 6, which is trained and evaluated on identical splits.*
