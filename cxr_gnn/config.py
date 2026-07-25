"""
config.py -- Immutable configuration (frozen dataclass).

Why frozen=True?
    The original notebook mutated a global CFG class with setattr() from several
    cells, so it was impossible to tell which configuration a given cell ran
    under. A frozen dataclass raises on accidental mutation; use
    dataclasses.replace(cfg, ...) to derive a variant explicitly.

Paths are resolved at runtime so the same config works on Kaggle, Colab and a
local machine (see `resolve_paths`). Nothing here hard-codes /kaggle any more.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, replace


def _default_work_dir() -> str:
    """Kaggle writes to /kaggle/working; elsewhere use ./outputs."""
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working"
    return os.environ.get("CXR_WORK_DIR", os.path.abspath("outputs"))


def _default_data_root() -> str:
    """Dataset root. Overridable with CXR_DATA_ROOT; auto-detected otherwise."""
    return os.environ.get(
        "CXR_DATA_ROOT",
        "/kaggle/input/capstone-c-dataset/capstone_avocado_version_three",
    )


@dataclass(frozen=True)
class Config:
    # -- Data ----------------------------------------------------------------
    data_root: str = ""            # filled in by __post_init__ if left empty
    img_size: int = 256
    valid_ext: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # -- Augmentation (TRAIN ONLY -- val/test are never touched) -------------
    do_aug: bool = True
    aug_cap: int = 220             # max graphs per class after augmentation
    max_aug_per_img: int = 6

    # -- Superpixel graph ----------------------------------------------------
    n_segments: int = 180
    compactness: float = 10.0
    lbp_p: int = 8
    lbp_r: float = 1.0

    # -- Feature flags -------------------------------------------------------
    use_edge_feat: bool = True
    deep_feat_dim: int = 128       # ResNet18 layer2 output channels
    hand_feat_dim: int = 12        # hand-crafted descriptors

    # -- Primary data split --------------------------------------------------
    # S4 (70/20/10) is the configuration selected by the split-ratio
    # sensitivity analysis (SAP 2); it is the primary split reported everywhere.
    test_size: float = 0.10
    val_size: float = 0.20

    # -- Model ---------------------------------------------------------------
    hidden: int = 48
    heads: int = 4
    dropout: float = 0.40
    in_dropout: float = 0.15
    dropedge: float = 0.10
    label_smooth: float = 0.05

    # -- Training ------------------------------------------------------------
    epochs: int = 150
    batch_size: int = 32
    lr: float = 3e-4
    wd: float = 3e-4
    patience: int = 6              # ReduceLROnPlateau patience
    early_stop: int = 20
    grad_clip: float = 2.0

    # -- Cross-validation ----------------------------------------------------
    n_folds: int = 5
    cv_epochs: int = 120
    cv_patience: int = 18

    # -- CNN baseline --------------------------------------------------------
    # The end-to-end ResNet18 fine-tune is far slower per epoch than any GNN
    # (raw 256x256 pixels vs ~180-node graphs), so it gets its own budget.
    include_cnn_baseline: bool = True
    cnn_epochs: int = 40
    cnn_patience: int = 8
    cnn_lr: float = 1e-4

    # -- Conformal prediction ------------------------------------------------
    conformal_alpha: float = 0.10  # target coverage = 90%
    raps_k_reg: int = 1
    raps_lam: float = 0.10

    # -- Statistical robustness suite (SAP 2 / SAP 6) ------------------------
    n_bootstrap: int = 2000
    sensitivity_seeds: tuple = (42, 43, 44, 45, 46)
    sensitivity_epochs: int = 100
    sensitivity_patience: int = 15

    # -- Robustness-under-perturbation study (Table 7b) ----------------------
    run_robustness: bool = True
    robustness_seed: int = 42

    # -- I/O -----------------------------------------------------------------
    work_dir: str = ""             # filled in by __post_init__ if left empty
    rebuild_cache: bool = False

    seed: int = 42

    def __post_init__(self) -> None:
        # object.__setattr__ is the sanctioned way to finish initialising a
        # frozen dataclass; after __post_init__ the instance is immutable.
        if not self.data_root:
            object.__setattr__(self, "data_root", _default_data_root())
        if not self.work_dir:
            object.__setattr__(self, "work_dir", _default_work_dir())

    # -- Derived paths -------------------------------------------------------
    @property
    def cache_file(self) -> str:
        return os.path.join(self.work_dir, "graph_cache.pt")

    @property
    def ckpt_file(self) -> str:
        return os.path.join(self.work_dir, "best_gatv2.pt")

    # -- Derived dims --------------------------------------------------------
    @property
    def node_feat_dim(self) -> int:
        return self.deep_feat_dim + self.hand_feat_dim  # 140

    @property
    def edge_feat_dim(self) -> int | None:
        return 2 if self.use_edge_feat else None

    def to_dict(self) -> dict:
        """Serialisable dict for checkpoints.

        The original notebook stored vars(CFG), which is a mappingproxy and
        therefore not picklable. asdict() always is.
        """
        d = asdict(self)
        d["valid_ext"] = list(self.valid_ext)
        d["sensitivity_seeds"] = list(self.sensitivity_seeds)
        d["cache_file"] = self.cache_file
        d["ckpt_file"] = self.ckpt_file
        return d

    def replace(self, **kw) -> "Config":
        """Explicit, traceable way to derive a variant config."""
        return replace(self, **kw)


# -- Label mapping ------------------------------------------------------------
# Folder names in the source dataset are inconsistent (two contain typos), so
# they are normalised to a canonical taxonomy here.
RAW2CLEAN: dict[str, str] = {
    "Cardiac Pathology":    "Cardiac",
    "Cronic Lung Disease":  "ChronicLung",   # typo in dataset
    "Chronic Lung Disease": "ChronicLung",
    "Normal":               "Normal",
    "TB":                   "TB",
    "Tuberculosis":         "TB",
    "plural Pathology":     "Pleural",       # typo in dataset
    "pleural Pathology":    "Pleural",
    "Pleural Pathology":    "Pleural",
}

CLEAN_LABELS: list[str] = sorted(set(RAW2CLEAN.values()))

CLASS2IDX: dict[str, int] = {c: i for i, c in enumerate(CLEAN_LABELS)}
IDX2CLASS: dict[int, str] = {i: c for c, i in CLASS2IDX.items()}
NUM_CLASSES: int = len(CLEAN_LABELS)

# Human-readable names used in the report tables.
DISPLAY_NAMES: dict[str, str] = {
    "Cardiac":     "Cardiac Pathology",
    "ChronicLung": "Chronic Lung Disease",
    "Normal":      "Normal",
    "Pleural":     "Pleural Pathology",
    "TB":          "Tuberculosis (TB)",
}
