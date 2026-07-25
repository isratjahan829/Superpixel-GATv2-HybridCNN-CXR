> ## NOT REPORTABLE
>
> These numbers come from **synthetic phantom images, not chest X-rays** and **reduced budgets** (epochs=8, cv_epochs=8, bootstrap B=300). They demonstrate that the pipeline executes end to end; they are **not findings** and must not be cited. Re-run on the real dataset at full budget to populate these tables.
>
> Source: `/home/user/Superpixel-GATv2-HybridCNN-CXR/demo_data` (81 images) - device cpu - commit `08d4c8a`

**Table 6. Model comparison: discrimination, agreement, calibration and cost.**

| Model | Acc. | Macro F1 | Macro AUC | Bal. Acc. | MCC (95% CI) | Cohen's kappa (95% CI) | Youden J | Brier | ECE | Log loss | Trainable params | Train (min/fold) | Infer. (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GCN | 0.361 | 0.388 | 0.816 | 0.467 | 0.333 (0.221-0.433) | 0.169 (0.074-0.281) | 0.296 | 0.768 | 0.103 | 1.533 | 41,997 | 0.01 | 1.7 |
| GraphSAGE | 0.167 | 0.195 | 0.852 | 0.400 | 0.206 (0.094-0.299) | 0.091 (0.031-0.158) | 0.218 | 0.810 | 0.198 | 1.649 | 78,093 | 0.01 | 1.3 |
| GAT | 0.083 | 0.031 | 0.735 | 0.200 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.000 | 0.824 | 0.256 | 1.671 | 43,197 | 0.01 | 2.2 |
| GATv2 Hybrid (ours) | 0.153 | 0.215 | 0.814 | 0.367 | 0.223 (0.106-0.295) | 0.076 (0.016-0.135) | 0.182 | 0.827 | 0.233 | 1.686 | 79,293 | 0.01 | 2.4 |
| ResNet18 (CNN baseline) | 0.319 | 0.185 | 0.639 | 0.267 | 0.094 (-0.015-0.214) | 0.071 (-0.011-0.161) | 0.081 | 0.790 | 0.094 | 1.585 | 11,179,077 | 0.05 | 10.3 |

MCC and Cohen's kappa stay meaningful under class imbalance, unlike raw accuracy. Brier score, ECE and log loss quantify how trustworthy the probabilities are, which any claim of clinical usefulness depends on. Parameter counts are TRAINABLE parameters: the GATv2 hybrid's ResNet18 encoder is frozen and therefore excluded, which is why its count is far below the fine-tuned CNN baseline's. Timings are for the hardware this run used and are not comparable across machines.
