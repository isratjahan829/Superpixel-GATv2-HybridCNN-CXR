"""
evaluation/stats.py -- Statistical toolkit for journal-grade reporting.

Everything the Statistical Analysis Plan asks for:
  - Non-parametric percentile bootstrap 95% CI (PAIRED resampling: the same
    indices index y_true and y_score, unlike a naive independent resample)
  - MCC, Cohen's kappa, Youden's J, multiclass Brier score, ECE/MCE, log loss
  - Wilson score interval (reliable at small n -- the minority classes here)
  - Positive/negative likelihood ratios
  - DeLong's test for paired ROC-AUC (fast O(n log n) form, Sun & Xu 2014),
    extended to multiclass one-vs-rest with both a Fisher-combined p-value and
    a macro z statistic
  - McNemar's exact test for paired accuracy
  - Holm-Bonferroni and Benjamini-Hochberg correction
  - Cohen's d for paired samples

Every function is pure numpy/scipy/sklearn: it operates on stored predictions,
so tables can be rebuilt without retraining anything.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm, chi2, ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from cxr_gnn.utils import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Bootstrap confidence intervals
# -----------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for an arbitrary metric.

    Both arrays are resampled with the SAME indices each iteration. Resampling
    them independently decorrelates predictions from labels and yields
    meaningless intervals -- a bug worth stating explicitly because it is easy
    to reintroduce.

    Args:
        y_true:    (n,) ground-truth labels
        y_score:   (n,) hard predictions or (n, n_classes) probabilities --
                   whatever metric_fn expects as its second argument
        metric_fn: callable(y_true_sample, y_score_sample) -> float
        n_boot:    number of resamples
        alpha:     1 - confidence level (0.05 -> 95% CI)

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = float(metric_fn(y_true[idx], y_score[idx]))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue          # degenerate resample (e.g. a class vanished)

    try:
        point = float(metric_fn(y_true, y_score))
    except Exception:
        point = float("nan")

    if len(vals) < max(10, n_boot // 20):
        logger.warning("Bootstrap produced only %d/%d valid resamples -- CI may be unreliable.",
                       len(vals), n_boot)
    if not vals:
        return point, float("nan"), float("nan")

    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def format_ci(point: float, lo: float, hi: float, decimals: int = 3) -> str:
    """'0.912 (0.881-0.940)' formatting for report tables."""
    if any(not np.isfinite(v) for v in (point, lo, hi)):
        return "n/a"
    fmt = f"%.{decimals}f"
    return f"{fmt % point} ({fmt % lo}-{fmt % hi})"


# -----------------------------------------------------------------------------
# Discrimination / calibration metrics
# -----------------------------------------------------------------------------

def macro_auc(y_true_idx: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """One-vs-rest macro AUC over the classes present in the sample."""
    y_true_idx = np.asarray(y_true_idx)
    yb = label_binarize(y_true_idx, classes=list(range(n_classes)))
    present = yb.sum(axis=0) > 0
    if present.sum() < 2:
        return float("nan")
    return float(roc_auc_score(yb[:, present], np.asarray(proba)[:, present], average="macro"))


def brier_score_multiclass(y_true_idx: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score: mean squared error against the one-hot label.

    Range [0, 2]; lower is better.
    """
    y_true_idx = np.asarray(y_true_idx)
    yb = label_binarize(y_true_idx, classes=list(range(n_classes))).astype(np.float64)
    if yb.shape[1] == 1:      # label_binarize collapses when only one class is present
        yb = np.hstack([1 - yb, yb])
    return float(np.mean(np.sum((np.asarray(proba) - yb) ** 2, axis=1)))


def multiclass_log_loss(y_true_idx: np.ndarray, proba: np.ndarray, n_classes: int,
                        eps: float = 1e-12) -> float:
    """Cross-entropy of the predicted distribution against the true label."""
    y_true_idx = np.asarray(y_true_idx).astype(int)
    p = np.clip(np.asarray(proba), eps, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y_true_idx)), y_true_idx])))


def youdens_j_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged Youden's J = sensitivity + specificity - 1, one-vs-rest."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    js = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        tn = np.sum((y_pred != c) & (y_true != c))
        fp = np.sum((y_pred == c) & (y_true != c))
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        js.append(sens + spec - 1)
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(js))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray,
                               n_bins: int = 10) -> tuple[float, float, dict]:
    """ECE / MCE from pooled (y_true, proba). Returns (ece, mce, bin_detail)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = proba.argmax(1)
    conf = proba.max(1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = mce = 0.0
    bin_acc, bin_conf, bin_cnt = [], [], []
    for b in range(n_bins):
        mask = (conf > bins[b]) & (conf <= bins[b + 1])
        if mask.sum() == 0:
            bin_acc.append(float("nan"))
            bin_conf.append((bins[b] + bins[b + 1]) / 2)
            bin_cnt.append(0)
            continue
        acc_b, conf_b = correct[mask].mean(), conf[mask].mean()
        gap = abs(acc_b - conf_b)
        bin_acc.append(float(acc_b))
        bin_conf.append(float(conf_b))
        bin_cnt.append(int(mask.sum()))
        ece += (mask.sum() / len(conf)) * gap
        mce = max(mce, gap)

    return float(ece), float(mce), {
        "bin_accuracy": bin_acc, "bin_confidence": bin_conf, "bin_count": bin_cnt,
    }


# -----------------------------------------------------------------------------
# Wilson intervals and per-class operating characteristics
# -----------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Preferred over the normal approximation at small n, which is exactly the
    regime of the minority classes here.
    """
    if n == 0:
        return float("nan"), float("nan")
    z = norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def per_class_sens_spec_ppv_npv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    class_names: Sequence[str],
    alpha: float = 0.05,
    proba: np.ndarray | None = None,
) -> dict:
    """One-vs-rest Sens/Spec/PPV/NPV with Wilson CIs, plus F1, LR+, LR- and
    (when `proba` is given) per-class AUC. This is the full row of the
    per-class performance table.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_total = len(y_true)
    out: dict = {}

    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        tn = int(np.sum((y_pred != c) & (y_true != c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))

        sens_n, spec_n, ppv_n, npv_n = tp + fn, tn + fp, tp + fp, tn + fn
        sens = _safe_div(tp, sens_n)
        spec = _safe_div(tn, spec_n)
        ppv = _safe_div(tp, ppv_n)
        npv = _safe_div(tn, npv_n)
        f1 = _safe_div(2 * tp, 2 * tp + fp + fn)

        # Likelihood ratios: LR+ = sens/(1-spec), LR- = (1-sens)/spec.
        lr_pos = _safe_div(sens, 1 - spec) if np.isfinite(spec) and spec < 1 else float("inf")
        lr_neg = _safe_div(1 - sens, spec) if np.isfinite(spec) and spec > 0 else float("nan")

        cls_auc = float("nan")
        if proba is not None:
            yb = (y_true == c).astype(int)
            if 0 < yb.sum() < len(yb):
                try:
                    cls_auc = float(roc_auc_score(yb, np.asarray(proba)[:, c]))
                except Exception:
                    cls_auc = float("nan")

        out[class_names[c]] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "support": sens_n,
            "prevalence": _safe_div(sens_n, n_total),
            "sensitivity": sens, "sensitivity_ci": wilson_ci(tp, sens_n, alpha),
            "specificity": spec, "specificity_ci": wilson_ci(tn, spec_n, alpha),
            "ppv": ppv, "ppv_ci": wilson_ci(tp, ppv_n, alpha),
            "npv": npv, "npv_ci": wilson_ci(tn, npv_n, alpha),
            "f1": f1, "auc": cls_auc,
            "lr_pos": lr_pos, "lr_neg": lr_neg,
        }
    return out


# -----------------------------------------------------------------------------
# DeLong's test
# -----------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance estimation (Sun & Xu 2014, Algorithm 2)."""
    n = preds_sorted_transposed.shape[1] - m
    pos = preds_sorted_transposed[:, :m]
    neg = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    return aucs, sx / m + sy / n


def delong_test(y_true_binary: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    """DeLong's test for two paired binary ROC-AUCs on the same instances.

    Returns dict(auc_a, auc_b, z, p); NaNs instead of raising on degenerate input.
    """
    y_true_binary = np.asarray(y_true_binary).astype(int)
    nan = {"auc_a": float("nan"), "auc_b": float("nan"), "z": float("nan"), "p": float("nan")}
    if y_true_binary.sum() in (0, len(y_true_binary)):
        return nan

    order = np.argsort(-y_true_binary, kind="mergesort")   # positives first
    m = int(y_true_binary.sum())
    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])

    try:
        aucs, cov = _fast_delong(preds, m)
    except Exception as ex:
        logger.warning("DeLong test failed: %s", ex)
        return nan

    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": 0.0, "p": 1.0}

    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z),
            "p": float(2 * (1 - norm.cdf(abs(z))))}


def delong_test_macro(
    y_true_idx: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    n_classes: int,
) -> dict:
    """Multiclass extension: one-vs-rest DeLong per class, combined two ways.

    - p_combined: Fisher's method over the per-class p-values (the headline
      significance test; its statistic is a chi-square, NOT a z).
    - z_macro / p_macro_z: Stouffer's weighted-free combination of the per-class
      z statistics, which keeps the DIRECTION of the difference. Report this one
      whenever the table has a signed "statistic" column.
    """
    y_true_idx = np.asarray(y_true_idx)
    per_class, pvals, zs = {}, [], []

    for c in range(n_classes):
        yb = (y_true_idx == c).astype(int)
        res = delong_test(yb, np.asarray(proba_a)[:, c], np.asarray(proba_b)[:, c])
        per_class[c] = res
        if np.isfinite(res["p"]):
            pvals.append(max(res["p"], 1e-300))       # avoid log(0)
        if np.isfinite(res["z"]):
            zs.append(res["z"])

    if not pvals:
        return {"per_class": per_class, "auc_a_macro": float("nan"),
                "auc_b_macro": float("nan"), "fisher_stat": float("nan"),
                "p_combined": float("nan"), "z_macro": float("nan"),
                "p_macro_z": float("nan")}

    stat = -2.0 * np.sum(np.log(pvals))
    p_combined = float(chi2.sf(stat, 2 * len(pvals)))

    # Stouffer: sum(z)/sqrt(k). Conservative here (the per-class tests are
    # correlated), so it is reported alongside Fisher rather than instead of it.
    z_macro = float(np.sum(zs) / np.sqrt(len(zs))) if zs else float("nan")
    p_macro_z = float(2 * (1 - norm.cdf(abs(z_macro)))) if zs else float("nan")

    with np.errstate(invalid="ignore"):
        auc_a_macro = float(np.nanmean([v["auc_a"] for v in per_class.values()]))
        auc_b_macro = float(np.nanmean([v["auc_b"] for v in per_class.values()]))

    return {
        "per_class": per_class,
        "auc_a_macro": auc_a_macro,
        "auc_b_macro": auc_b_macro,
        "fisher_stat": float(stat),
        "p_combined": p_combined,
        "z_macro": z_macro,
        "p_macro_z": p_macro_z,
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on paired correctness vectors.

    The two prediction vectors MUST come from the same instances in the same
    order, otherwise the discordant counts are meaningless.
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    n01 = int(np.sum(a_ok & ~b_ok))    # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))    # A wrong, B right
    table = np.array([[int(np.sum(a_ok & b_ok)), n01],
                      [n10, int(np.sum(~a_ok & ~b_ok))]])

    try:
        from statsmodels.stats.contingency_tables import mcnemar as _mcnemar
        p = float(_mcnemar(table, exact=True).pvalue)
    except Exception:
        # Exact binomial fallback, so the suite never depends on statsmodels.
        from scipy.stats import binomtest
        n = n01 + n10
        p = 1.0 if n == 0 else float(binomtest(min(n01, n10), n, 0.5).pvalue)

    stat = float((abs(n01 - n10) - 1) ** 2 / (n01 + n10)) if (n01 + n10) else float("nan")
    return {"table": table.tolist(), "n01": n01, "n10": n10,
            "chi2_cc": stat, "p": p,
            "acc_a": float(a_ok.mean()), "acc_b": float(b_ok.mean())}


# -----------------------------------------------------------------------------
# Multiple-comparison correction
# -----------------------------------------------------------------------------

def holm_bonferroni(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down correction, returned in the ORIGINAL order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(max((m - rank) * pvals[idx], prev), 1.0)
        corrected[idx] = adj
        prev = adj
    return corrected


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction, returned in the ORIGINAL order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    corrected_sorted = ranked * m / (np.arange(m) + 1)
    corrected_sorted = np.minimum.accumulate(corrected_sorted[::-1])[::-1]
    corrected = np.empty(m)
    corrected[order] = np.clip(corrected_sorted, 0, 1)
    return corrected


# -----------------------------------------------------------------------------
# Effect size and paired tests
# -----------------------------------------------------------------------------

def cohens_d_paired(x: Sequence[float], y: Sequence[float]) -> float:
    """Cohen's d for paired samples: mean(diff) / sd(diff)."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    sd = diff.std(ddof=1)
    return 0.0 if sd == 0 else float(diff.mean() / sd)


def interpret_d(d: float) -> str:
    """Conventional Cohen's-d labels, for the effect-size column."""
    a = abs(d)
    if not np.isfinite(a):
        return "n/a"
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def paired_significance(x: Sequence[float], y: Sequence[float]) -> dict:
    """Paired t-test + Wilcoxon signed-rank + Cohen's d for two matched vectors
    (e.g. accuracy across matching seeds for split config A vs B).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = {"mean_diff": float(np.mean(x - y)), "cohens_d": cohens_d_paired(x, y)}
    out["effect_size_label"] = interpret_d(out["cohens_d"])

    if np.allclose(x, y):
        # Both tests are undefined when every difference is exactly zero.
        out["t_stat"], out["t_p"], out["wilcoxon_p"] = 0.0, 1.0, 1.0
        return out

    try:
        r = ttest_rel(x, y)
        out["t_stat"], out["t_p"] = float(r.statistic), float(r.pvalue)
    except Exception:
        out["t_stat"], out["t_p"] = float("nan"), float("nan")
    try:
        out["wilcoxon_p"] = float(wilcoxon(x, y).pvalue)
    except Exception:
        out["wilcoxon_p"] = float("nan")
    return out
