### TRIPOD-AI checklist

| Item | Status | Evidence |
|---|---|---|
| Title/abstract identifies the study as developing an AI/ML prediction model | needs manual entry | Add when writing up. |
| Source of data and eligibility criteria described | needs manual entry | Document the source institution(s) and inclusion criteria. |
| Outcome (target classes) clearly defined | addressed | 5-class taxonomy fixed in config.py (CLASS2IDX). |
| Predictors (features) fully specified | addressed | Hand-crafted + deep node features documented in graph.py / encoder.py. |
| Sample size / cases per class justified or acknowledged as a limitation | addressed | `split_sensitivity.json` -- Split-ratio sensitivity plus per-class support. |
| Missing / unusable data handling described | addressed | `run.log` -- Degenerate-segmentation drops are logged per split by cache.py. |
| Development vs validation data separation stated | addressed | `split_sensitivity.json` -- Image-level stratified split before augmentation. |
| Internal validation method reported | addressed | `cv_results.json` -- 5-fold CV plus bootstrap 95% CIs. |
| Performance measures justified for the clinical task and class imbalance | addressed | `statistical_robustness.json` -- Macro-F1, balanced accuracy, MCC, kappa. |
| Calibration reported | addressed | `calibration.json` -- ECE/MCE, reliability diagram, temperature scaling. |
| Model updating / re-calibration discussed | N/A | Out of scope for a retrospective single-institution study. |
| Comparison against existing models / baselines | addressed | `baseline_results.json` -- GCN/GraphSAGE/GAT/ResNet18 plus literature table. |
| Uncertainty quantification for individual predictions | addressed | `conformal_results.json` -- Conformal prediction sets with coverage guarantees. |
| External validation on an independent dataset | N/A | Single-source data only -- stated as a limitation. |
| Code / model availability statement | addressed | This repository; add a DOI at publication time. |

### STARD-AI checklist

| Item | Status | Evidence |
|---|---|---|
| Study design (retrospective / prospective) stated | needs manual entry | State explicitly in Methods. |
| Reference standard (ground-truth labelling process) described | needs manual entry | Document the radiologist/clinical labelling protocol. |
| Flow of images (inclusion, exclusions, degenerate cases) reported | addressed | `run.log` -- Per-split drop counts logged by cache.py. |
| Distribution of severity / alternative diagnoses in the sample | needs manual entry | Add clinical characterisation if available. |
| Statistical methods pre-specified and matched to the data | addressed | `statistical_robustness.json` -- Wilson CIs at small n, non-parametric bootstrap, Holm/FDR correction. |
| Indeterminate results handling | addressed | `conformal_results.json` -- Non-singleton conformal sets flag exactly these. |
| Adverse events / harms of testing discussed | N/A | Not applicable to retrospective image classification. |
