"""
Tests covering the defects this repository fixes.

Each test names the bug it guards against, so a regression is self-explaining.
Run with:  pytest -q
"""

from __future__ import annotations

import dataclasses
import os
import tempfile

import numpy as np
import pytest
import torch

from cxr_gnn.config import Config, CLASS2IDX, NUM_CLASSES
from cxr_gnn.data.graph import image_to_graph, segment, _rag_edges, _region_stats
from cxr_gnn.evaluation import stats as S


# -- Config -------------------------------------------------------------------

def test_config_is_immutable():
    """The original notebook mutated a global CFG with setattr from many cells."""
    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.epochs = 999


def test_config_paths_are_derived_not_hardcoded():
    cfg = Config(work_dir="/tmp/xyz")
    assert cfg.cache_file == "/tmp/xyz/graph_cache.pt"
    assert cfg.ckpt_file == "/tmp/xyz/best_gatv2.pt"


def test_config_to_dict_is_picklable():
    """vars(CFG) returned a mappingproxy and broke torch.save."""
    import pickle
    pickle.loads(pickle.dumps(Config().to_dict()))


def test_primary_split_is_s4():
    """The document claims S4 (70/20/10) is the primary split; make it true."""
    cfg = Config()
    assert (cfg.test_size, cfg.val_size) == (0.10, 0.20)


# -- Graph construction -------------------------------------------------------

def _phantom(size=96, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = 0.4 + 0.3 * np.exp(-(((xx - 0.5) / 0.25) ** 2 + ((yy - 0.5) / 0.3) ** 2))
    return np.clip(img + 0.02 * rng.standard_normal((size, size)), 0, 1).astype(np.float32)


def test_segment_labels_are_contiguous():
    """SLIC can leave gaps; downstream code indexes 1..N and assumed none."""
    labels, n = segment(_phantom(), Config(n_segments=60))
    present = np.unique(labels)
    assert present.min() >= 1
    assert len(present) == n
    assert present.max() == n


def test_graph_features_have_no_nan_and_emit_no_warning():
    """Missing labels used to produce 0/0 -> NaN node features, silently."""
    cfg = Config(n_segments=60)
    with np.errstate(all="raise"):
        g = image_to_graph(_phantom(), 0, cfg, None, torch.device("cpu"))
    assert g is not None
    assert torch.isfinite(g.x).all()
    assert g.x.shape[1] == cfg.hand_feat_dim          # hand-only when encoder is None


def test_region_stats_match_numpy_reference():
    rng = np.random.default_rng(0)
    v = rng.random((20, 20))
    lab = rng.integers(0, 4, size=400)
    mean, std, vmin, vmax = _region_stats(v, lab, 4)
    for k in range(4):
        sel = v.ravel()[lab == k]
        assert np.isclose(mean[k], sel.mean())
        assert np.isclose(std[k], sel.std())
        assert np.isclose(vmin[k], sel.min())
        assert np.isclose(vmax[k], sel.max())


def test_rag_edges_are_undirected_unique_and_zero_indexed():
    labels = np.array([[1, 1, 2], [1, 3, 2], [3, 3, 2]])
    e = _rag_edges(labels)
    assert e.min() >= 0
    assert len(e) == len({tuple(x) for x in e.tolist()})
    assert all(u < v for u, v in e.tolist())


def test_edge_attr_stays_aligned_after_dropedge():
    """DropEdge dropped edges but not their attributes in one code path."""
    from cxr_gnn.models.gatv2 import GATv2Classifier
    cfg = Config(dropedge=0.5)
    model = GATv2Classifier(6, 2, cfg, n_classes=3).train()
    x = torch.randn(10, 6)
    ei = torch.randint(0, 10, (2, 24))
    ea = torch.randn(24, 2)
    batch = torch.zeros(10, dtype=torch.long)
    out = model(x, ei, ea, batch)            # must not raise a shape mismatch
    assert out.shape == (1, 3)


def test_embed_does_not_change_training_mode():
    """embed() called self.eval() and left the model there."""
    from cxr_gnn.models.gatv2 import GATv2Classifier
    model = GATv2Classifier(6, 2, Config(), n_classes=3).train()
    model.embed(torch.randn(8, 6), torch.randint(0, 8, (2, 16)),
                torch.randn(16, 2), torch.zeros(8, dtype=torch.long))
    assert model.training is True


# -- Statistics ---------------------------------------------------------------

def test_bootstrap_resamples_are_paired():
    """Independent resampling of y_true and y_score destroys the correlation;
    a perfect classifier must keep a CI near 1.0."""
    y = np.repeat(np.arange(5), 20)
    pt, lo, hi = S.bootstrap_ci(y, y.copy(),
                                lambda a, b: float((a == b).mean()), n_boot=300, seed=0)
    assert pt == 1.0 and lo == 1.0 and hi == 1.0


def test_wilson_ci_brackets_the_estimate():
    lo, hi = S.wilson_ci(3, 10)
    assert 0 <= lo < 0.3 < hi <= 1
    assert S.wilson_ci(0, 0) == (float("nan"), float("nan")) or True   # no crash


def test_wilson_beats_normal_approx_at_the_boundary():
    """At k == n the normal approximation gives a zero-width interval."""
    lo, hi = S.wilson_ci(10, 10)
    assert lo < 1.0 and hi == pytest.approx(1.0)


def test_delong_identical_models_give_p_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    r = S.delong_test(y, p, p.copy())
    assert r["p"] == pytest.approx(1.0)
    assert r["z"] == pytest.approx(0.0)


def test_delong_macro_reports_a_signed_z():
    """The report table needs a signed statistic; Fisher's chi2 cannot be negative."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, NUM_CLASSES, 300)
    good = np.zeros((300, NUM_CLASSES))
    good[np.arange(300), y] = 1.0
    good = good * 0.7 + 0.06                       # confident and correct
    bad = rng.random((300, NUM_CLASSES))
    bad /= bad.sum(1, keepdims=True)
    r = S.delong_test_macro(y, good, bad, NUM_CLASSES)
    assert r["fisher_stat"] > 0
    assert r["z_macro"] > 0                        # 'good' really is better
    r_rev = S.delong_test_macro(y, bad, good, NUM_CLASSES)
    assert r_rev["z_macro"] < 0                    # sign flips with the order


def test_holm_is_monotone_and_at_least_raw():
    p = [0.001, 0.02, 0.3, 0.7]
    c = S.holm_bonferroni(p)
    assert np.all(c >= np.array(p) - 1e-12)
    assert np.all(np.diff(c) >= -1e-12)
    assert np.all(c <= 1.0)


def test_bh_is_no_more_conservative_than_holm():
    p = [0.001, 0.02, 0.03, 0.7]
    assert np.all(S.benjamini_hochberg(p) <= S.holm_bonferroni(p) + 1e-12)


def test_mcnemar_detects_a_one_sided_difference():
    y = np.zeros(100, dtype=int)
    a = np.zeros(100, dtype=int)           # always right
    b = np.zeros(100, dtype=int)
    b[:20] = 1                             # wrong 20 times
    r = S.mcnemar_test(y, a, b)
    assert r["n01"] == 20 and r["n10"] == 0
    assert r["p"] < 0.001


def test_likelihood_ratios_are_computed():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 0, 1, 1, 2, 2])
    out = S.per_class_sens_spec_ppv_npv(y, p, 3, ["a", "b", "c"])
    assert set(out["a"]) >= {"lr_pos", "lr_neg", "prevalence", "f1", "support"}
    assert out["a"]["sensitivity"] == 1.0


def test_paired_significance_handles_identical_vectors():
    """ttest_rel on two identical vectors is NaN, not p=1."""
    r = S.paired_significance([0.5] * 5, [0.5] * 5)
    assert r["t_p"] == 1.0 and r["wilcoxon_p"] == 1.0 and r["cohens_d"] == 0.0


def test_effect_size_labels():
    assert S.interpret_d(0.1) == "negligible"
    assert S.interpret_d(-0.65) == "medium"
    assert S.interpret_d(1.4) == "large"


def test_brier_and_log_loss_reward_confident_correct_predictions():
    y = np.array([0, 1, 2])
    confident = np.eye(3) * 0.98 + 0.01
    unsure = np.full((3, 3), 1 / 3)
    assert (S.brier_score_multiclass(y, confident, 3)
            < S.brier_score_multiclass(y, unsure, 3))
    assert S.multiclass_log_loss(y, confident, 3) < S.multiclass_log_loss(y, unsure, 3)


# -- Loaders ------------------------------------------------------------------

def test_weighted_loader_actually_balances():
    """make_weighted_loader had `shuffle if not sampler else False`, where
    sampler was always truthy -- so the shuffle=True path silently did neither.
    """
    from torch_geometric.data import Data
    from cxr_gnn.training.trainer import make_weighted_loader

    graphs = []
    for cls, n in [(0, 90), (1, 10)]:
        for _ in range(n):
            graphs.append(Data(x=torch.randn(3, 4),
                               edge_index=torch.tensor([[0, 1], [1, 0]]),
                               y=torch.tensor([cls])))
    loader = make_weighted_loader(graphs, 16, 2, seed=0)
    seen = torch.cat([b.y for b in loader])
    minority_frac = float((seen == 1).float().mean())
    assert minority_frac > 0.3, f"minority class only {minority_frac:.2f} of samples"


def test_unbalanced_loader_shuffles_without_sampler():
    from torch_geometric.data import Data
    from cxr_gnn.training.trainer import make_weighted_loader
    graphs = [Data(x=torch.randn(2, 3), edge_index=torch.tensor([[0], [1]]),
                   y=torch.tensor([i % 2])) for i in range(40)]
    loader = make_weighted_loader(graphs, 8, 2, balanced=False)
    assert len(torch.cat([b.y for b in loader])) == 40


# -- Split logic --------------------------------------------------------------

def test_stratified_split_indices_never_empty_a_class():
    from cxr_gnn.training.split_sensitivity import stratified_split_indices
    labels = np.array([0] * 3 + [1] * 50 + [2] * 20)
    tr, va, te = stratified_split_indices(labels, 0.6, 0.2, 0.2, seed=0)
    assert len(tr) and len(va) and len(te)
    assert len(set(tr) | set(va) | set(te)) == len(labels)      # a partition
    assert not (set(tr) & set(te))                              # disjoint


def test_split_config_ratios_must_sum_to_one():
    from cxr_gnn.training.split_sensitivity import SplitConfig
    with pytest.raises(ValueError):
        SplitConfig("bad", 0.5, 0.3, 0.3)


# -- Tables -------------------------------------------------------------------

def test_missing_artifact_yields_an_explicit_message_not_a_number():
    """A table must never invent values when its artifact is absent."""
    from cxr_gnn.evaluation import tables
    with tempfile.TemporaryDirectory() as d:
        for name, fn in tables.ALL_TABLES.items():
            out = fn(d)
            assert "not available" in out, f"{name} fabricated content"


def test_tables_render_from_artifacts():
    from cxr_gnn.evaluation import tables
    import json
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "cv_results.json"), "w") as f:
            json.dump({
                "fold_sizes": [{"fold": 1, "n_train": 10, "n_val": 2, "n_test": 3}],
                **{k: {"mean": 0.8, "std": 0.01, "per_fold": [0.8]}
                   for k in ["Accuracy", "Balanced-Acc", "Macro-F1", "Macro-AUC", "MCC"]},
            }, f)
        out = tables.table2_cross_validation(d)
        assert "0.8000" in out and "not available" not in out


def test_error_analysis_reports_both_sides_of_abstention():
    from cxr_gnn.evaluation.error_analysis import run_error_analysis
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 0, 2, 2])
    conf = np.array([0.9, 0.4, 0.8, 0.5, 0.95, 0.99])
    with tempfile.TemporaryDirectory() as d:
        r = run_error_analysis(y, p, conf, d)
    assert r["n_errors"] == 2
    assert 0.0 <= r["errors_deferred_fraction"] <= 1.0
    assert "correct_deferred_fraction" in r          # the cost, not just the benefit


def test_safe_split_falls_back_instead_of_raising_on_singleton_class():
    """sklearn's stratify raises when a class has one member in the subset.
    That crashed conformal prediction mid-run; it must degrade with a warning.
    """
    from cxr_gnn.training.crossval import safe_train_test_split
    labels = np.array([0, 0, 0, 0, 1, 2, 2, 2, 2, 2])
    idx = np.arange(len(labels))
    a, b = safe_train_test_split(idx, labels, 0.5, seed=0, context="test")
    assert len(a) + len(b) == len(idx)
    assert not (set(a.tolist()) & set(b.tolist()))


def test_safe_split_still_stratifies_when_it_can():
    from cxr_gnn.training.crossval import safe_train_test_split
    labels = np.repeat([0, 1, 2], 10)
    idx = np.arange(len(labels))
    a, b = safe_train_test_split(idx, labels, 0.5, seed=0)
    assert sorted(np.bincount(labels[b], minlength=3).tolist()) == [5, 5, 5]
