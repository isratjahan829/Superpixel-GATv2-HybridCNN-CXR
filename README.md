# Superpixel GATv2 + Hybrid CNN — Chest X-Ray Classification

Five-class chest X-ray classification (Cardiac, Chronic Lung, Normal, Pleural,
Tuberculosis) with a hybrid pipeline: a **frozen ResNet18** supplies deep
features, SLIC **superpixels** turn each image into a region-adjacency graph,
and a **GATv2** network classifies the graph.

Dataset: <https://www.kaggle.com/datasets/shakib0hasan/capstone-c-dataset>

```
image ──► SLIC (~180 superpixels) ──► graph
              │                         nodes: 128 deep (frozen ResNet18, region-pooled)
              │                                + 12 hand-crafted (intensity, LBP, shape)
              │                         edges: region adjacency, 2 features
              └──► GATv2 ×2 ──► mean-pool ‖ max-pool ──► MLP ──► 5 classes
```

## Quick start

```bash
pip install -r requirements.txt

# Kaggle: the dataset is auto-detected under /kaggle/input
python train.py

# Anywhere else
CXR_DATA_ROOT=/path/to/capstone_avocado_version_three python train.py

# Verify the pipeline runs, in a few minutes, without the real data
python tools/make_demo_dataset.py --out demo_data --per-class 26 --size 128
CXR_DATA_ROOT=demo_data python train.py --fast --work-dir outputs_demo
```

Selected stages only:

```bash
python train.py --only test cv          # train, evaluate, cross-validate
python train.py --skip ablation robustness
python train.py --rebuild-cache         # after changing any graph setting
```

## Notebook

`notebooks/gatv2_cxr_pipeline.ipynb` runs the same pipeline step by step with
figures and tables inline. It **imports** `cxr_gnn` rather than carrying its own
copy of the source, so the notebook and the tested package cannot drift apart.
Regenerate it after changing the driver:

```bash
python tools/build_notebook.py
```

If the real dataset is absent, the notebook generates synthetic phantom images
and runs anyway — clearly labelled, because metrics on phantoms are a smoke
test, not findings.

## Repository layout

```
cxr_gnn/
  config.py               frozen dataclass; paths resolved at runtime
  utils.py                seeding, device, logging, parameter counting
  data/
    dataset.py            discovery, loading, image-level stratified split
    augment.py            medically-safe augmentation (no flips — see below)
    graph.py              SLIC → region-adjacency graph, node/edge features
    cache.py              graph cache with a config signature
  models/
    encoder.py            frozen ResNet18 feature extractor
    gatv2.py              GATv2 classifier (+ single-head ablation variant)
    cnn_baseline.py       end-to-end fine-tuned ResNet18
  training/
    trainer.py            training loop; eval-mode metrics for both splits
    crossval.py           5-fold CV, model comparison, pairwise statistics
    ablation.py           feature arms and architecture arms
    split_sensitivity.py  5 split ratios × N seeds
  evaluation/
    stats.py              bootstrap, Wilson, DeLong, McNemar, Holm/BH, effect sizes
    conformal.py          LAC / APS / RAPS / class-conditional LAC
    calibration.py        ECE / MCE + temperature scaling
    robustness.py         performance under input perturbation
    error_analysis.py     misclassification taxonomy, abstention trade-off
    reporting.py          literature table, TRIPOD-AI / STARD-AI checklists
    tables.py             rebuilds every report table from saved artifacts
    visualization.py      all figures
train.py                  single entry point
tools/                    notebook generator, synthetic dataset generator
tests/                    tests, one per fixed defect
docs/                     review of the evaluation document
```

## Design decisions worth knowing

**No horizontal or vertical flips.** A horizontally flipped chest X-ray shows
the heart on the right, which is the radiographic presentation of dextrocardia —
the augmentation would change the diagnosis, not the nuisance parameters.
Only rotation, translation, zoom and intensity changes are applied.

**Split before augmentation.** The image-level split happens on raw files.
Building all graphs first and splitting afterwards puts augmented copies of
training images into the test set.

**Train metrics measured in `eval()` mode.** Measuring them during the
optimisation pass (dropout and DropEdge active) makes training accuracy look
worse than validation accuracy, which reads as impossible.

**One set of fold indices, shared by every model.** McNemar's and DeLong's tests
are *paired*: they are meaningless unless each model was evaluated on exactly
the same held-out instances. A guard refuses the test when the label vectors
differ.

**The frozen encoder has no trainable parameters.** The GATv2 hybrid trains
~79k parameters; the encoder's 683k are frozen. Comparisons against the
11.2M-parameter fine-tuned CNN baseline should say which count they mean.

**Artifacts, not transcription.** Each stage writes JSON; `evaluation/tables.py`
rebuilds every report table from those files. A missing artifact yields
"not available — run step X", never a placeholder number.

## Outputs

| File | Contents |
|---|---|
| `test_results.json` | held-out test metrics |
| `cv_results.json` | per-fold CV metrics including MCC |
| `ablation_features.json`, `ablation_architecture.json` | ablation arms |
| `baseline_results.json` | per-model fold metrics, parameters, timings |
| `statistical_robustness.json` | model-level metrics, per-class Wilson CIs, all pairwise tests |
| `split_sensitivity.json` | per-config metrics, CIs, pooled predictions, confusion matrix |
| `conformal_results.json` | coverage and set size per method |
| `calibration.json` | ECE/MCE, bins, temperature scaling |
| `robustness.json` | accuracy under each perturbation |
| `error_analysis.json` | misclassification taxonomy |
| `evaluation_tables.md` | every table, regenerated |
| `fig1..fig11*.png` | figures |
| `run.log` | full run log |

## Tests

```bash
pytest -q
```

Each test names the defect it guards against — paired bootstrap resampling,
DropEdge/edge-attribute alignment, the weighted sampler actually balancing,
signed DeLong statistics, tables refusing to invent numbers, and so on.

## Reviewing the evaluation document

`docs/EVALUATION_TABLES_REVIEW.md` checks every table in
`Evaluation_Tables_FINAL_1.docx` against the code and the stored run log, listing
what did not match, what had no code behind it, and where two tables contradicted
each other. The central finding of that document — that the fine-tuned CNN
baseline outperforms the hybrid GNN, and the graph variants are statistically
indistinguishable — holds up and is reproduced by this pipeline.

## Licence

MIT — see `LICENSE`.
