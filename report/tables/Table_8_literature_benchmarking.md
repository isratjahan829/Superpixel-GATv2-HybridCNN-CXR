> ## NOT REPORTABLE
>
> These numbers come from **synthetic phantom images, not chest X-rays** and **reduced budgets** (epochs=8, cv_epochs=8, bootstrap B=300). They demonstrate that the pipeline executes end to end; they are **not findings** and must not be cited. Re-run on the real dataset at full budget to populate these tables.
>
> Source: `/home/user/Superpixel-GATv2-HybridCNN-CXR/demo_data` (81 images) - device cpu - commit `08d4c8a`

**Table 8. Literature benchmarking (qualitative positioning).**

| Study / Model | Year | Dataset (n) | Task / Classes | Approach | Acc. | Macro AUC | Macro F1 | Sens. (macro) |
|---|---|---|---|---|---|---|---|---|
| CheXNet (Rajpurkar et al., 2017) | 2017 | ChestX-ray14 (112,120 images) | 14 findings, multi-label binary | 121-layer DenseNet | n/a (multi-label) | ~0.83-0.84 mean AUROC | 0.435 (pneumonia) | n/a |
| COVID-Net (Wang, Lin & Wong, 2020) | 2020 | COVIDx (~13,975 images) | 3-class multiclass | Lightweight CNN (PEPX) | 0.933 | n/a | n/a | ~0.91 (COVID class) |
| MIMIC-CXR single-source benchmark | recent | MIMIC-CXR (~377,110 images) | Multi-label findings | CNN multi-label classifier | n/a | ~0.75 mean AUROC | n/a | n/a |
| **GATv2 Hybrid (this work)** | this study | This dataset (81) | 5-class multiclass | Frozen ResNet18 features + superpixel GATv2 | 0.444 | 0.943 | 0.489 | 0.600 |

*Because label sets, task framing and dataset scale differ by two to three orders of magnitude, this table supports only relative discussion (direction and size of the gap, with the dataset-scale caveat), never a claim of superiority. The only controlled head-to-head comparison in this work is the CNN baseline in Table 6, which is trained and evaluated on identical splits.*
