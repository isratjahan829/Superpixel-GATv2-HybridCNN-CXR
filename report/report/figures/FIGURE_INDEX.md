# Figure index

**Figure_01_training_curves.png** -- Training and validation loss/accuracy, and the generalisation gap. Both curves are measured in eval() mode, so they are directly comparable.

**Figure_02_confusion_test_set.png** -- Confusion matrix on the held-out test set, counts and row-normalised.

**Figure_03_confusion_best_split.png** -- Confusion matrix for the best split configuration, pooled over seeds. This is the matrix behind Table 4.

**Figure_04_per_class_metrics.png** -- Per-class precision, recall and F1 on the test set.

**Figure_05_roc_and_precision_recall.png** -- One-vs-rest ROC and precision-recall curves with per-class AUC/AP.

**Figure_06_tsne_embedding.png** -- t-SNE of the graph-level embeddings before the classification head.

**Figure_07_superpixel_graphs.png** -- SLIC superpixel graphs per class: nodes are regions, edges are adjacency.

**Figure_08_attention_saliency.png** -- Attention-based saliency: where the GATv2 layer places its attention mass.

**Figure_09a_ablation_features.png** -- Feature ablation: hand-crafted vs deep vs hybrid node features.

**Figure_09b_ablation_architecture.png** -- Architecture ablation: one design choice removed per arm (Table 7a).

**Figure_10_model_comparison.png** -- Macro-F1 and macro-AUC across GCN, GraphSAGE, GAT, GATv2 and the CNN baseline.

**Figure_11_calibration.png** -- Reliability diagram and confidence histogram (ECE/MCE).

**Figure_12_robustness.png** -- Accuracy under each input perturbation against the clean reference (Table 7b).
