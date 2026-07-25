"""
evaluation/visualization.py -- Training curves, confusion matrix, ROC/PR,
t-SNE, superpixel graphs, attention saliency, ablation and comparison bars,
reliability diagram.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend; safe on Kaggle
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    precision_recall_fscore_support, confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries

from cxr_gnn.config import Config, IDX2CLASS, NUM_CLASSES, RAW2CLEAN
from cxr_gnn.data.dataset import load_gray
from cxr_gnn.data.graph import image_to_graph, segment, _rag_edges
from cxr_gnn.utils import get_logger

logger = get_logger(__name__)
COLORS = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))


def _savefig(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.debug("Saved %s", path)
    return path


# -- Training curves ----------------------------------------------------------

def plot_training_curves(history: dict, work_dir: str) -> str:
    ep = range(1, len(history["tr_loss"]) + 1)
    best_ep = int(np.argmin(history["va_loss"]) + 1)
    gap = np.array(history["tr_acc"]) - np.array(history["va_acc"])

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    ax[0].plot(ep, history["tr_loss"], label="train")
    ax[0].plot(ep, history["va_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(ep, history["tr_acc"], label="train acc")
    ax[1].plot(ep, history["va_acc"], label="val acc")
    ax[1].plot(ep, history["va_bacc"], label="val bal-acc")
    ax[1].axhline(max(history["va_acc"]), ls="--", c="gray", alpha=0.5,
                  label=f"best={max(history['va_acc']):.3f}")
    ax[1].set_ylim(0.0, 1.02)
    ax[1].set_title("Accuracy (both measured in eval mode)")
    ax[1].legend(fontsize=8)

    ax[2].plot(ep, gap, color="crimson")
    ax[2].axhline(0, c="k", lw=0.8)
    ax[2].fill_between(ep, gap, 0, color="crimson", alpha=0.15)
    ax[2].set_title("Generalisation gap (train - val acc)")

    for a in ax:
        a.axvline(best_ep, ls=":", c="green", alpha=0.7)
        a.set_xlabel("epoch")

    return _savefig(fig, os.path.join(work_dir, "fig1_training_curves.png"))


# -- Confusion matrix ---------------------------------------------------------

def plot_confusion(ys, ps, target_names, work_dir: str,
                   fname: str = "fig2_confusion.png") -> str:
    cm = confusion_matrix(ys, ps, labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(1, keepdims=True)
    # A class with no support would otherwise produce 0/0 and a blank row.
    cm_norm = np.divide(cm, np.clip(row_sums, 1, None), dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k, (mat, ttl) in enumerate([(cm, "Counts"), (cm_norm, "Row-normalised")]):
        im = ax[k].imshow(mat, cmap="Blues")
        ax[k].set_title(f"Confusion ({ttl})")
        ax[k].set_xticks(range(NUM_CLASSES))
        ax[k].set_yticks(range(NUM_CLASSES))
        ax[k].set_xticklabels(target_names, rotation=45, ha="right")
        ax[k].set_yticklabels(target_names)
        ax[k].set_xlabel("Predicted")
        ax[k].set_ylabel("True")
        vmax = mat.max() if mat.max() > 0 else 1
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                v = mat[i, j]
                ax[k].text(j, i, f"{v:.2f}" if k else f"{int(v)}", ha="center", va="center",
                           color="white" if v > vmax * 0.6 else "black")
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    return _savefig(fig, os.path.join(work_dir, fname))


# -- Per-class metric bars ----------------------------------------------------

def plot_per_class_metrics(ys, ps, target_names, work_dir: str) -> str:
    prec, rec, f1c, _ = precision_recall_fscore_support(
        ys, ps, labels=list(range(NUM_CLASSES)), zero_division=0)
    xx = np.arange(NUM_CLASSES)
    bw = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(xx - bw, prec, bw, label="precision")
    ax.bar(xx, rec, bw, label="recall")
    ax.bar(xx + bw, f1c, bw, label="F1")
    ax.set_xticks(xx)
    ax.set_xticklabels(target_names, rotation=20)
    ax.set_ylim(0, 1.08)
    ax.set_title("Per-class precision / recall / F1")
    ax.legend()
    return _savefig(fig, os.path.join(work_dir, "fig3_per_class_metrics.png"))


# -- ROC and PR curves --------------------------------------------------------

def plot_roc_pr(ys, probs, target_names, work_dir: str) -> str:
    y_bin = label_binarize(ys, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    aucs = {}
    for i in range(NUM_CLASSES):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        aucs[i] = auc(fpr, tpr)
        ax[0].plot(fpr, tpr, color=COLORS[i], lw=2, label=f"{target_names[i]} AUC={aucs[i]:.3f}")
    fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), probs.ravel())
    ax[0].plot(fpr_m, tpr_m, "k--", lw=2, label=f"micro AUC={auc(fpr_m, tpr_m):.3f}")
    ax[0].plot([0, 1], [0, 1], c="gray", ls=":")
    macro = np.mean(list(aucs.values())) if aucs else float("nan")
    ax[0].set_title(f"ROC (macro-AUC={macro:.3f})")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].legend(fontsize=9)

    for i in range(NUM_CLASSES):
        if y_bin[:, i].sum() == 0:
            continue
        pr, rc, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        apv = average_precision_score(y_bin[:, i], probs[:, i])
        ax[1].plot(rc, pr, color=COLORS[i], lw=2, label=f"{target_names[i]} AP={apv:.3f}")
    ax[1].set_title("Precision-Recall")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].legend(fontsize=9)
    return _savefig(fig, os.path.join(work_dir, "fig4_roc_pr.png"))


# -- t-SNE --------------------------------------------------------------------

@torch.no_grad()
def plot_tsne(model, train_loader, test_loader, device, work_dir: str,
              target_names=None) -> str:
    target_names = target_names or [IDX2CLASS[i] for i in range(NUM_CLASSES)]
    model.eval()

    def _embed(loader):
        E, Y = [], []
        for b in loader:
            b = b.to(device)
            h = model.embed(b.x, b.edge_index, b.edge_attr, b.batch)
            E.append(h.cpu())
            Y.append(b.y.cpu())
        return torch.cat(E).numpy(), torch.cat(Y).numpy()

    tr_emb, tr_y = _embed(train_loader)
    te_emb, te_y = _embed(test_loader)
    all_emb = np.concatenate([tr_emb, te_emb])
    all_y = np.concatenate([tr_y, te_y])
    is_test = np.concatenate([np.zeros(len(tr_y)), np.ones(len(te_y))]).astype(bool)

    # perplexity must stay below n_samples; clamp for small runs.
    perp = float(max(2, min(30, (len(all_emb) - 1) // 3)))
    emb2d = TSNE(n_components=2, perplexity=perp, init="pca",
                 learning_rate="auto", random_state=42).fit_transform(all_emb)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for i in range(NUM_CLASSES):
        m = all_y == i
        ax[0].scatter(emb2d[m, 0], emb2d[m, 1], s=14, color=COLORS[i], alpha=0.55,
                      label=target_names[i])
    ax[0].set_title(f"t-SNE (perplexity={perp:.0f})")
    ax[0].legend(fontsize=9)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].scatter(emb2d[~is_test, 0], emb2d[~is_test, 1], s=10, c="lightgray",
                  alpha=0.4, label="train")
    for i in range(NUM_CLASSES):
        m = is_test & (all_y == i)
        ax[1].scatter(emb2d[m, 0], emb2d[m, 1], s=70, color=COLORS[i],
                      edgecolors="black", linewidths=0.6, label=f"test:{target_names[i]}")
    ax[1].set_title("Test samples in embedding space")
    ax[1].legend(fontsize=8, ncol=2)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    return _savefig(fig, os.path.join(work_dir, "fig5_tsne.png"))


# -- Superpixel graph ---------------------------------------------------------

def plot_superpixel_graphs(samples, cfg: Config, target_names, work_dir: str) -> str:
    by_class = defaultdict(list)
    for path, raw in samples:
        by_class[RAW2CLEAN.get(raw, raw)].append(path)

    keys = [IDX2CLASS[i] for i in range(NUM_CLASSES)]
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(4 * NUM_CLASSES, 4))
    for ax, key, title in zip(np.atleast_1d(axes), keys, target_names):
        if not by_class[key]:
            ax.axis("off")
            continue
        img = load_gray(sorted(by_class[key])[0], cfg.img_size)
        labels, _n = segment(img, cfg)
        ax.imshow(mark_boundaries(np.stack([img] * 3, -1), labels, color=(1, 1, 0)))
        props = regionprops(labels)
        cent = {p.label: p.centroid for p in props}
        for u, v in _rag_edges(labels) + 1:      # back to 1-indexed label ids
            if u in cent and v in cent:
                (y1, x1), (y2, x2) = cent[u], cent[v]
                ax.plot([x1, x2], [y1, y2], lw=0.4, color="red", alpha=0.5)
        for p in props:
            ax.plot(p.centroid[1], p.centroid[0], "o", ms=2, color="cyan")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.suptitle("SLIC superpixel graphs (nodes = regions, edges = adjacency)")
    return _savefig(fig, os.path.join(work_dir, "fig6_superpixel_graphs.png"))


# -- Attention saliency -------------------------------------------------------

@torch.no_grad()
def plot_attention_saliency(model, samples, cfg: Config, target_names, encoder,
                            device, work_dir: str, n_examples: int = 3) -> str:
    """Overlay layer-2 attention mass per superpixel.

    The `encoder` argument is required: the original version passed None, so
    the graphs carried 12 hand-crafted features into a model expecting 140 and
    every forward pass raised. The exception was swallowed by a bare except and
    the figure came out blank -- a silent failure worth removing the cause of.
    """
    if encoder is None:
        raise ValueError(
            "plot_attention_saliency needs the same encoder the model was trained with; "
            "passing None produces feature-dimension mismatches on every image.")

    by_class = defaultdict(list)
    for path, raw in samples:
        by_class[RAW2CLEAN.get(raw, raw)].append(path)

    model.eval()
    keys = [IDX2CLASS[i] for i in range(NUM_CLASSES)]
    fig, ax_grid = plt.subplots(NUM_CLASSES, n_examples,
                                figsize=(3.2 * n_examples, 3.2 * NUM_CLASSES),
                                squeeze=False)

    n_failed = 0
    for r, (key, title) in enumerate(zip(keys, target_names)):
        paths = sorted(by_class[key])[:n_examples]
        for c in range(n_examples):
            ax = ax_grid[r][c]
            ax.axis("off")
            if c >= len(paths):
                continue

            img = load_gray(paths[c], cfg.img_size)
            labels, n_nodes = segment(img, cfg)
            g = image_to_graph(img, 0, cfg, encoder, device)
            if g is None:
                n_failed += 1
                continue

            g = g.to(device)
            bt = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
            out, (ei, att) = model(g.x, g.edge_index, g.edge_attr, bt,
                                   return_attention=True)

            pred = target_names[int(out.argmax(1))]
            att = att.mean(1).cpu().numpy()
            dst = ei[1].cpu().numpy()
            sc = np.zeros(n_nodes)
            np.add.at(sc, dst, att)
            if sc.max() > 0:
                sc = sc / sc.max()
            sal = sc[labels - 1]                 # label map -> per-pixel saliency

            ax.imshow(img, cmap="gray")
            ax.imshow(sal, cmap="jet", alpha=0.45)
            ax.set_title(f"{title} [{'ok' if pred == title else '-> ' + pred}]", fontsize=8)

    if n_failed:
        logger.warning("Attention saliency: %d example(s) had degenerate segmentations.", n_failed)
    plt.suptitle("Attention-based saliency (where GATv2 attends)", y=1.001)
    return _savefig(fig, os.path.join(work_dir, "fig7_attention_saliency.png"))


# -- Ablation and comparison bars ---------------------------------------------

def plot_ablation(ablation_results: dict, work_dir: str,
                  fname: str = "fig8_ablation.png") -> str:
    names = list(ablation_results.keys())
    f1s = [float(np.mean(ablation_results[n]["f1"])) for n in names]
    f1e = [float(np.std(ablation_results[n]["f1"])) for n in names]
    aucs = [float(np.nanmean(ablation_results[n]["auc"])) for n in names]

    xx = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    ax[0].bar(xx, f1s, yerr=f1e, capsize=4, color="#4caf72")
    for i, (v, e) in enumerate(zip(f1s, f1e)):
        ax[0].text(i, v + e + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax[0].set_title("Ablation: Macro-F1")
    ax[0].set_ylim(0, 1.05)

    ax[1].bar(xx, aucs, color="#3b7dd8")
    for i, v in enumerate(aucs):
        if np.isfinite(v):
            ax[1].text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    ax[1].set_title("Ablation: Macro-AUC")
    finite = [v for v in aucs if np.isfinite(v)]
    ax[1].set_ylim(max(0.0, min(finite) - 0.1) if finite else 0.0, 1.02)

    for a in ax:
        a.set_xticks(xx)
        a.set_xticklabels(names, rotation=20, fontsize=7, ha="right")
    plt.tight_layout()
    return _savefig(fig, os.path.join(work_dir, fname))


def plot_model_comparison(model_results: dict, work_dir: str) -> str:
    order = list(model_results.keys())
    f1s = [float(np.mean(model_results[m]["f1"])) for m in order]
    aucs = [float(np.nanmean(model_results[m]["auc"])) for m in order]
    xx = np.arange(len(order))
    # One colour per model, however many there are (this was hardcoded to 4 and
    # broke when the CNN baseline made it 5).
    cmap = plt.get_cmap("tab10")
    cols = [cmap(i % 10) for i in range(len(order))]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for j, (vals, ttl) in enumerate([(f1s, "Macro-F1"), (aucs, "Macro-AUC")]):
        finite = [v for v in vals if np.isfinite(v)]
        lo = max(0.0, min(finite) - 0.1) if finite else 0.0
        ax[j].bar(xx, vals, color=cols)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax[j].text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
        ax[j].set_xticks(xx)
        ax[j].set_xticklabels(order, rotation=20, fontsize=8, ha="right")
        ax[j].set_title(ttl)
        ax[j].set_ylim(lo, 1.02)
    plt.tight_layout()
    return _savefig(fig, os.path.join(work_dir, "fig9_model_comparison.png"))


# -- Reliability diagram ------------------------------------------------------

def plot_calibration(cal_result: dict, work_dir: str) -> str:
    n = cal_result["n_bins"]
    bins = np.linspace(0, 1, n + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    ba = np.array([x if x is not None else float("nan") for x in cal_result["bin_accuracy"]])
    bc = np.array(cal_result["bin_count"], dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot([0, 1], [0, 1], "k--", label="perfect")
    ax[0].bar(centers, np.nan_to_num(ba), width=1 / n, edgecolor="black",
              alpha=0.75, color="#4caf72", label="model")
    for c, a in zip(centers, ba):
        if np.isfinite(a):
            ax[0].plot([c, c], [a, c], color="crimson", lw=1, alpha=0.5)
    ax[0].set_title(f"Reliability diagram (ECE={cal_result['ECE']:.3f})")
    ax[0].set_xlabel("confidence (max softmax)")
    ax[0].set_ylabel("accuracy")
    ax[0].legend()
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)

    total = bc.sum() if bc.sum() > 0 else 1.0
    ax[1].bar(centers, bc / total, width=1 / n, color="#3b7dd8", edgecolor="black")
    ax[1].axvline(cal_result["avg_confidence"], ls="--", c="orange",
                  label=f"avg conf={cal_result['avg_confidence']:.2f}")
    ax[1].axvline(cal_result["oof_accuracy"], ls="--", c="green",
                  label=f"OOF acc={cal_result['oof_accuracy']:.2f}")
    ax[1].set_title("Confidence histogram")
    ax[1].set_xlabel("confidence")
    ax[1].set_ylabel("fraction")
    ax[1].legend()
    return _savefig(fig, os.path.join(work_dir, "fig10_calibration.png"))


def plot_robustness(robustness: dict, work_dir: str) -> str:
    rows = [r for r in robustness["rows"] if not r.get("skipped")]
    names = [r["perturbation"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    xx = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["#3b7dd8"] + ["#4caf72"] * (len(names) - 1)
    ax.bar(xx, accs, color=colors)
    ax.axhline(robustness["clean_reference"]["accuracy"], ls="--", c="crimson",
               label="clean reference")
    for i, r in enumerate(rows):
        ax.text(i, r["accuracy"] + 0.005, f"{r['accuracy']:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xx)
    ax.set_xticklabels(names, rotation=25, fontsize=7, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy under input perturbation")
    ax.legend()
    plt.tight_layout()
    return _savefig(fig, os.path.join(work_dir, "fig11_robustness.png"))
