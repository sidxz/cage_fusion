# CAGEFusion — Pretraining + OpenADMET Benchmark Plan

**Status:** Planning complete, implementation not started.
**Branch:** refactor4
**Goal:** (1) Pretrain CAGEFusion backbone on broad public ADMET data, push weights to
`cage-fusion/cage-fusion-pretrained` on HuggingFace. (2) Fine-tune on the official
OpenADMET ExpansionRx benchmark and compare MA-RAE against leaderboard.

---

## 1. Background & Motivation

The OpenADMET ExpansionRx challenge uses a time-split of ~5,330 train / 2,280 test
molecules across 9 ADMET endpoints. 5,330 molecules is borderline for CAGEFusion's
multi-modal fusion weights (graph encoder + cross-attention) which are cold-started.
ChemBERTa (77M param language model) is already pretrained, but the fusion components need
prior knowledge of ADMET-space structure.

Solution: two-stage training.
- Stage 1 — broad ADMET pretraining on TDC + MoleculeNet (~50k–150k labelled pairs)
- Stage 2 — fine-tune on ExpansionRx using backbone from Stage 1

After pretraining the backbone transfers to ANY downstream task (classification or
regression) via `strict=False` loading — nuisance compound detection, single-endpoint
permeability, hERG, etc.

---

## 2. Architecture Changes Required

### 2a. `cage_fusion/modeling/modeling_cage.py`

**Add to `CAGEFusionPreTrainedModel` (base class):**

```python
def freeze_backbone(self) -> None:
    """Freeze all encoder weights. Only the task head remains trainable."""
    for name, param in self.named_parameters():
        if not any(name.startswith(h) for h in ("classifier", "regressor")):
            param.requires_grad_(False)

def unfreeze_backbone(self) -> None:
    """Unfreeze all parameters (call after freeze phase)."""
    for param in self.parameters():
        param.requires_grad_(True)

def save_backbone(self, save_directory: str) -> None:
    """Save only encoder weights (no head). Safe for cross-task loading."""
    os.makedirs(save_directory, exist_ok=True)
    encoder_state = {
        k: v for k, v in self.state_dict().items()
        if not k.startswith(("classifier.", "regressor."))
    }
    torch.save(encoder_state, os.path.join(save_directory, "backbone.bin"))
    self.config.save_pretrained(save_directory)
    logger.info("Saved backbone to %s", save_directory)
```

**Fix masked MSE in `CAGEFusionForRegression.forward()`:**

Replace current `nn.MSELoss()(predictions, labels)` with NaN-aware version:

```python
if labels is not None:
    mask = ~torch.isnan(labels)
    if mask.any():
        diff = (predictions - labels.nan_to_num(0.0)) ** 2
        mse  = (diff * mask).sum() / mask.sum().clamp(min=1)
    else:
        mse = predictions.sum() * 0.0   # zero loss, no valid targets
    loss = mse + lambda_entropy * enc.attn_entropy_loss + lambda_prior * enc.token_prior_loss
```

### 2b. `cage_fusion/training/training_args.py`

Add two new fields to `TrainingArguments`:

```python
primary_metric: str = "rmse"
# Options: "rmse", "mae", "r2", "marae", "auc", "mcc"
# Controls which metric selects best_model.pt

primary_metric_direction: str = "min"
# "min" (lower is better: rmse, mae, marae) or "max" (higher: r2, auc, mcc)
```

Note: `max_grad_norm` already exists — no gradient clipping change needed.

### 2c. `cage_fusion/training/metrics.py`

Add `MARAEAccumulator`:

```python
class MARAEAccumulator:
    """
    Macro-Averaged Relative Absolute Error — matches OpenADMET leaderboard formula.

    Per-endpoint RAE_i = MAE_i / mean(|y_true_i - mean(y_true_i)|)
    MA-RAE = mean(RAE_i across tasks with >= 2 valid samples)

    NaN targets in labels are masked per-task (sparse multi-task support).
    """
    def __init__(self, num_tasks, label_names=None): ...
    def update(self, targets_batch, preds_batch): ...  # stores raw (pred, target) pairs; masks NaN
    def compute(self, reduce="mean"):
        # returns (ma_rae_float, per_task_rae_list)
    def reset(self): ...
```

### 2d. `cage_fusion/training/trainer.py`

- Import `MARAEAccumulator`
- In `train_epoch()` and `evaluate()`: when `self.task == "regression"`, also compute MA-RAE
  alongside RMSE/MAE/R² using `MARAEAccumulator`
- Extend `_empty_history()` to include `"train_marae"` / `"val_marae"` keys for regression
- Change checkpoint selection logic to use `args.primary_metric` +
  `args.primary_metric_direction` instead of hardcoded RMSE comparison:

```python
p_key  = args.primary_metric   # e.g. "marae" or "rmse"
p_dir  = args.primary_metric_direction  # "min" or "max"
p_val  = val_metrics[p_key]
if p_dir == "min":
    p_improved = p_val < best_primary
else:
    p_improved = p_val > best_primary
```

---

## 3. New Script Files

```
scripts/
  pretrain_admet.py          ← Stage 1: broad ADMET pretraining
  finetune_openadmet.py      ← Stage 2: ExpansionRx fine-tuning + eval
  evaluate_openadmet.py      ← Load checkpoint, write MA-RAE report + submission CSV

benchmarks/openadmet/
  __init__.py
  preprocessing.py           ← log-transforms, inverse-transform, column mapping
  marae.py                   ← standalone MA-RAE scorer (matches leaderboard formula)
  data_loader.py             ← TDC + HF dataset fetching helpers
  README.md
```

---

## 4. Stage 1 — Broad ADMET Pretraining (`scripts/pretrain_admet.py`)

### 4a. Data sources

Fetched via `pytdc` (TDC Python package) and HuggingFace `datasets`:

| Dataset | Source | Endpoint | ~N molecules | Transform |
|---|---|---|---|---|
| Lipophilicity_AstraZeneca | TDC | LogD | 4,200 | none (already log) |
| Solubility_AqSolDB | TDC | logS | 9,982 | none |
| ESOL | MoleculeNet | logS | 1,128 | none |
| FreeSolv | MoleculeNet | ΔGhyd | 642 | none |
| Lipophilicity | MoleculeNet | logD | 4,200 | none |
| Caco2_Wang | TDC | log Papp | 906 | none (already log10) |
| PAMPA_NCATS | TDC | log Papp | 2,035 | none |
| PPBR_AZ | TDC | % unbound | 1,797 | log1p |
| VDss_Lombardo | TDC | log VDss | 1,130 | none |
| HalfLife_Obach | TDC | log t½ | 667 | none |
| Clearance_Hepatocyte_AZ | TDC | log CLint | 1,020 | none |
| Clearance_Microsome_AZ | TDC | log CLint | 1,102 | none |
| hERG | TDC | pIC50 | 648 | none |
| DILI | TDC | DILI label | 475 | none (regression score) |

**Total: ~28,000 unique molecules, ~50,000–80,000 (molecule, endpoint) pairs**

All datasets are merged into a wide DataFrame with SMILES + one column per endpoint.
Molecules not measured for a given endpoint have NaN — masked MSE handles this naturally.

### 4b. Pretraining config

```python
config = CageFusionConfig(
    num_labels=14,              # one per TDC/MoleculeNet endpoint
    model_task="regression",
    label_names=[...],          # endpoint names
    attn_mode="cross",
    hidden_size=128,            # FIXED — all downstream fine-tuning must match
    num_heads=8,
    co_attention_layers=2,
    cross_attn_dropout=0.15,
    proj_dropout=0.10,
    fusion_dropout_1=0.3,
    fusion_dropout_2=0.2,
    use_fg_prompt=True,
)
```

**CRITICAL**: `hidden_size=128` is committed here. All fine-tuning tasks must use the
same backbone width. Changing it requires repretraining.

### 4c. Training args

```python
args = TrainingArguments(
    output_dir="runs/pretrain_admet",
    checkpoints_dir="checkpoints/pretrain_admet",
    num_epochs=100,
    batch_size=64,
    learning_rate=3e-4,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    warmup_fraction=0.05,
    primary_metric="rmse",           # NEW FIELD
    primary_metric_direction="min",  # NEW FIELD
    seed=42,
)
```

### 4d. Script flow

```
1. Load each TDC/MoleculeNet dataset via data_loader.py
2. Apply per-dataset forward_transform() from preprocessing.py
3. Merge all into a single wide DataFrame (SMILES, endpoint_1, ..., endpoint_14)
   — NaN where molecule was not measured for that endpoint
4. Random 85/15 train/val split (random OK for pretraining, no time-split needed)
5. CageFusionDataModule.from_dataframes(train_df, val_df, label_cols=ALL_ENDPOINTS)
6. Build model: CAGEFusionForRegression(config)
7. trainer.train()  ← masked MSE handles NaN targets automatically
8. model.save_backbone("checkpoints/pretrain_admet/")
   model.save_pretrained("checkpoints/pretrain_admet/")
9. Push to HuggingFace: CageFusionPipeline.push_to_hub(
       "checkpoints/pretrain_admet/",
       repo_id="cage-fusion/cage-fusion-pretrained",
       model="best",
   )
```

---

## 5. Stage 2 — ExpansionRx Fine-tuning (`scripts/finetune_openadmet.py`)

### 5a. Dataset

```python
from datasets import load_dataset
train_ds = load_dataset("openadmet/openadmet-expansionrx-challenge-data", split="train")
test_ds  = load_dataset("openadmet/openadmet-expansionrx-challenge-data", split="test")
train_df = train_ds.to_pandas()  # 5,330 rows
test_df  = test_ds.to_pandas()   # 2,280 rows
```

Official train/test split is time-based — must be respected. Do NOT re-split.
Carve a random 15% validation set from train_df only (for hyperparameter / early stopping).

### 5b. Column names (exact, from HF dataset)

```python
LABEL_COLS = [
    "LogD",
    "KSOL",
    "HLM_CLint",
    "MLM_CLint",
    "Caco-2_Permeability_Papp_A_B",
    "Caco-2_Permeability_Efflux",
    "MPPB",
    "MBPB",
    "MGMB",
]
SMILES_COL    = "SMILES"
NAME_COL      = "Molecule_Name"
```

### 5c. Log transforms (from `benchmarks/openadmet/preprocessing.py`)

```python
TRANSFORM_TABLE = {
    "LogD":                         ("none",  1.0),    # already log-scale
    "KSOL":                         ("log1p", 1e-6),   # µM → dimensionless
    "HLM_CLint":                    ("log1p", 1.0),
    "MLM_CLint":                    ("log1p", 1.0),
    "Caco-2_Permeability_Papp_A_B": ("log1p", 1e-6),   # 10⁻⁶ cm/s
    "Caco-2_Permeability_Efflux":   ("log1p", 1.0),
    "MPPB":                         ("log1p", 1.0),
    "MBPB":                         ("log1p", 1.0),
    "MGMB":                         ("log1p", 1.0),
}

def forward_transform(df):
    """Apply per-endpoint log transforms. Returns new DataFrame."""
    ...

def inverse_transform(arr, cols):
    """Invert transforms for submission-scale predictions."""
    ...
```

`forward_transform` is applied to label columns of both train and test DataFrames
before building the data module. `inverse_transform` is applied to model predictions
in the evaluation script before writing `submission.csv`.

### 5d. Fine-tuning config

```python
finetune_config = CageFusionConfig(
    num_labels=9,
    model_task="regression",
    label_names=LABEL_COLS,
    hidden_size=128,        # ← must match pretrained backbone
    attn_mode="cross",
    # all other arch params same as pretraining
)
```

### 5e. Fine-tuning flow

```
Phase A — head-only warmup (5 epochs):
  model = CAGEFusionForRegression.from_pretrained(
      "cage-fusion/cage-fusion-pretrained",
      config=finetune_config,
      strict=False,       # ignore pretrained head (14 labels → 9 labels mismatch)
  )
  model.freeze_backbone()
  trainer_A = Trainer(model, args_phase_A, ...)  # lr=1e-3, 5 epochs
  trainer_A.train()

Phase B — full fine-tuning:
  model.unfreeze_backbone()
  trainer_B = Trainer(model, args_phase_B, ...)  # lr=3e-4, 50 epochs, primary_metric="marae"
  trainer_B.train()
  model.save_pretrained("checkpoints/openadmet_finetuned/")
```

Phase A trains only the new 9-label head. Phase B unfreezes everything and jointly
refines backbone + head with MA-RAE as checkpoint selection metric.

### 5f. Training args

```python
# Phase A
args_A = TrainingArguments(
    checkpoints_dir="checkpoints/openadmet_phaseA",
    num_epochs=5,
    learning_rate=1e-3,
    weight_decay=1e-4,
    primary_metric="rmse",
    primary_metric_direction="min",
    seed=42,
)

# Phase B
args_B = TrainingArguments(
    checkpoints_dir="checkpoints/openadmet_phaseB",
    num_epochs=60,
    learning_rate=3e-4,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    primary_metric="marae",           # leaderboard metric
    primary_metric_direction="min",
    seed=42,
)
```

---

## 6. Evaluation (`scripts/evaluate_openadmet.py`)

### 6a. MA-RAE formula (`benchmarks/openadmet/marae.py`)

```python
def compute_marae(y_true: np.ndarray, y_pred: np.ndarray,
                  label_names: list) -> dict:
    """
    Matches OpenADMET leaderboard formula exactly.

    y_true, y_pred: (N, 9) arrays in LOG SCALE (same scale as training).
    Missing values (NaN in y_true) are masked per-task.

    Returns dict with keys:
      "ma_rae"           : float (primary leaderboard metric)
      "per_endpoint"     : {name: {"mae", "rae", "r2", "spearman", "kendall"}}
    """
    from scipy.stats import spearmanr, kendalltau
    raes = []
    per_ep = {}
    for i, name in enumerate(label_names):
        mask = ~np.isnan(y_true[:, i])
        yt   = y_true[mask, i]
        yp   = y_pred[mask, i]
        if len(yt) < 2:
            continue
        mae  = np.mean(np.abs(yt - yp))
        rae  = mae / np.mean(np.abs(yt - np.mean(yt)))
        r2   = r2_score(yt, yp)
        spr  = spearmanr(yt, yp).statistic
        ktau = kendalltau(yt, yp).statistic
        raes.append(rae)
        per_ep[name] = {"mae": mae, "rae": rae, "r2": r2, "spearman": spr, "kendall": ktau}
    return {"ma_rae": float(np.mean(raes)), "per_endpoint": per_ep}
```

### 6b. Evaluation script flow

```
1. Load best checkpoint: AutoCageFusion.from_pretrained("checkpoints/openadmet_phaseB/")
2. Run on official test_df (log-scale labels + SMILES)
3. Collect (preds, true_labels) in log scale
4. compute_marae(true_labels, preds, LABEL_COLS) → print full report
5. Print per-endpoint table:
   | Endpoint                     | MAE   | RAE   | R²    | Spearman |
   |------------------------------|-------|-------|-------|----------|
   | LogD                         | 0.xxx | 0.xxx | 0.xxx | 0.xxx    |
   ...
   | MA-RAE (ours)                |       | 0.xxx |       |          |
   | MA-RAE (leaderboard #1)      |       | 0.511 |       |          |
6. Inverse-transform predictions → original scale → write submission.csv
   Columns: Molecule_Name, LogD, KSOL, HLM_CLint, ..., MGMB
7. Write evaluation_report.csv with per-endpoint metrics
```

---

## 7. Competitive Enhancements (implement after baseline works)

In priority order:

| Enhancement | Expected gain | Effort |
|---|---|---|
| Ensemble 5–10 seeds, average predictions | ~5–10% MA-RAE reduction | Low |
| Raw dataset: use `>X` modifiers as censored regression | Moderate | Medium |
| Retrain on full train set (no val) for final submission | Small (~15% more data) | Low |
| SMILES enumeration augmentation | Moderate | Medium |
| Larger model: `hidden_size=256` (needs re-pretraining) | Unknown | High |

---

## 8. HuggingFace Upload

After pretraining:
```python
CageFusionPipeline.push_to_hub(
    "checkpoints/pretrain_admet/",
    repo_id="cage-fusion/cage-fusion-pretrained",
    model="best",
    commit_message="Broad ADMET pretraining: 14 endpoints, ~28k molecules, TDC+MoleculeNet",
)
```

Repo tags: `cage-fusion-pretrained`, `admet`, `multi-task`, `regression`
Includes: `pytorch_model.bin`, `backbone.bin`, `config.json`, `training_args.json`

When loading for fine-tuning:
```python
model = CAGEFusionForRegression.from_pretrained(
    "cage-fusion/cage-fusion-pretrained",
    config=my_finetune_config,
    strict=False,
)
model.freeze_backbone()
```

---

## 9. File Change Summary

| File | Type | Change |
|---|---|---|
| `cage_fusion/modeling/modeling_cage.py` | Modify | Add `freeze_backbone()`, `unfreeze_backbone()`, `save_backbone()` to base class; masked MSE in `CAGEFusionForRegression` |
| `cage_fusion/training/training_args.py` | Modify | Add `primary_metric` + `primary_metric_direction` fields |
| `cage_fusion/training/metrics.py` | Modify | Add `MARAEAccumulator` class |
| `cage_fusion/training/trainer.py` | Modify | Use `primary_metric`/`direction` for checkpoint selection; compute MA-RAE in evaluate() for regression |
| `scripts/pretrain_admet.py` | New | Stage 1 pretraining script |
| `scripts/finetune_openadmet.py` | New | Stage 2 fine-tuning + phased training |
| `scripts/evaluate_openadmet.py` | New | MA-RAE report + submission CSV |
| `benchmarks/openadmet/__init__.py` | New | Package init |
| `benchmarks/openadmet/preprocessing.py` | New | Log transforms + inverse |
| `benchmarks/openadmet/marae.py` | New | Standalone MA-RAE scorer |
| `benchmarks/openadmet/data_loader.py` | New | TDC + HF dataset fetching |
| `benchmarks/openadmet/README.md` | New | Usage instructions |

---

## 10. Implementation Order

1. Source changes first (needed by all scripts):
   a. `modeling_cage.py` — masked MSE + backbone helpers
   b. `training_args.py` — new fields
   c. `metrics.py` — MARAEAccumulator
   d. `trainer.py` — configurable primary metric

2. Benchmark infrastructure:
   a. `benchmarks/openadmet/preprocessing.py`
   b. `benchmarks/openadmet/marae.py`
   c. `benchmarks/openadmet/data_loader.py`

3. Scripts:
   a. `scripts/pretrain_admet.py`
   b. `scripts/finetune_openadmet.py`
   c. `scripts/evaluate_openadmet.py`

4. Verify: run evaluate_openadmet.py on test set, compare MA-RAE to leaderboard

---

## 11. Key Decisions / Constraints

- `hidden_size=128` is the canonical backbone width. Do NOT change without full re-pretraining.
- The official OpenADMET test split is time-based and must not be re-randomised.
- MA-RAE is computed in LOG SCALE (same scale as training targets). Do NOT inverse-transform before computing MA-RAE.
- `submission.csv` contains INVERSE-TRANSFORMED predictions (original measurement units).
- `backbone.bin` excludes head weights; `pytorch_model.bin` includes them.
- `strict=False` in `from_pretrained` is required whenever switching task heads.
