# OpenADMET ExpansionRx Benchmark

End-to-end pipeline to train CAGEFusion on the OpenADMET ExpansionRx dataset and
compare MA-RAE performance against the published leaderboard.

## Quick start

```bash
# 1. Pretrain backbone on broad ADMET data (~28k molecules, TDC + MoleculeNet)
python scripts/pretrain_admet.py

# 2. Fine-tune on the official OpenADMET ExpansionRx split
python scripts/finetune_openadmet.py

# 3. Evaluate on the official test set and compare to leaderboard
python scripts/evaluate_openadmet.py --plot
```

## Dataset

| Split | Molecules | Source |
|-------|-----------|--------|
| Train | 5,330 | `openadmet/openadmet-expansionrx-challenge-data` (HF, default config) |
| Test  | 2,280 | Same dataset, time-split (official, do not re-randomise) |

9 endpoints: LogD, KSOL, HLM_CLint, MLM_CLint, Caco-2_Permeability_Papp_A_B,
Caco-2_Permeability_Efflux, MPPB, MBPB, MGMB.

## Metric

**MA-RAE** (Macro-Averaged Relative Absolute Error) — lower is better.

```
RAE_i  = MAE_i / mean(|y_true_i - mean(y_true_i)|)
MA-RAE = mean(RAE_i)   over endpoints with >= 2 valid samples
```

Leaderboard #1 score: **0.5113** (as of January 2026).

## Directory layout on /data-1

```
/data-1/
  cage-fusion-pretrain/
    datasets/          ← TDC + MoleculeNet CSVs + pretrain_merged.csv
    checkpoints/       ← pretrained backbone.bin + pytorch_model.bin
    features/          ← HDF5 feature caches for pretraining
    logs/              ← training curves + history CSV

  cage-fusion-admet/
    datasets/          ← OpenADMET HF cache
    checkpoints/       ← fine-tuned model (best by MA-RAE)
    features/          ← HDF5 feature caches for fine-tuning
    logs/              ← training curves
    submissions/
      evaluation_report.csv
      submission.csv         ← predictions in original measurement units
      scatter_plots/
```

## Transfer learning to other tasks

After pretraining, the backbone can be loaded for ANY downstream task:

```python
from cage_fusion import CageFusionConfig
from cage_fusion.modeling.modeling_cage import CAGEFusionForMultiLabelClassification

config = CageFusionConfig(
    num_labels=4,
    model_task="classification",
    label_names=["PAINS_A", "Aggregator", "hERG", "Promiscuous"],
    hidden_size=128,   # must match pretrained backbone
    attn_mode="cross",
)
model = CAGEFusionForMultiLabelClassification.from_pretrained(
    "/data-1/cage-fusion-pretrain/checkpoints/",
    config=config,
    strict=False,   # head shape mismatch is expected and handled
)
model.freeze_backbone()     # phase A: train only the new head
# ... train ...
model.unfreeze_backbone()   # phase B: full fine-tuning
```
