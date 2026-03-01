# Plan: Clean Transfer Learning API

## Goals

1. **`from_backbone()`** — clean classmethod for cross-task transfer (encoder loaded,
   head randomly initialized, warning printed).
2. **`from_pretrained()` override** — same-task fine-tuning; shows load report box.
3. **`save_backbone()` fix** — exclude D-MPNN's internal task-specific predictor from
   `backbone.bin` so it is truly task-agnostic and never causes shape mismatches.
4. **`push_to_hub` / `save_pretrained`** — always write `backbone.bin` alongside
   `pytorch_model.bin`; both go to the same HF repo.
5. **Scripts** — replace manual `torch.load` + shape-filter blocks with the new API.

---

## Background: why the shape mismatch happens

ChemProp's `MPNN` embeds a task-specific predictor head internally under
`encoder.graph_encoder._mpnn.predictor.*` and `encoder.graph_encoder._mpnn.metrics.*`.
These tensors are sized by `num_labels`. They are currently saved into `backbone.bin`
and cause `RuntimeError` when the source and target tasks have different label counts.

The fix is to add these prefixes to the exclusion list in `save_backbone()`.

---

## Design Decisions

- **HF repo**: single repo (`cage-fusion/cage-fusion-pretrained`) with two files:
  `pytorch_model.bin` (full model + head) and `backbone.bin` (encoder only, safe for
  any num_labels).
- **Box**: shown on every checkpoint load — both `from_pretrained` and `from_backbone`.
- **Warning**: printed only when `from_backbone` is used (head is randomly initialized).
- **API location**: new methods live on `CAGEFusionPreTrainedModel` (base class) and are
  re-exported via `AutoCageFusion`.

---

## Files to Change (in order)

### 1. `cage_fusion/modeling/modeling_cage.py`

#### 1a. Fix `save_backbone()`

Extend `_HEAD_PREFIXES` to also exclude the D-MPNN's internal task-specific layers:

```python
_HEAD_PREFIXES: tuple = (
    "classifier.",
    "regressor.",
    # ChemProp's internal task head — sized by num_labels, not transferable
    "encoder.graph_encoder._mpnn.predictor.",
    "encoder.graph_encoder._mpnn.metrics.",
)
```

This makes `backbone.bin` universally loadable regardless of num_labels.

#### 1b. Add `_load_report()` static helper

A private static method that accepts `state_dict, missing, unexpected, shape_skipped,
model` and prints the rich box. Shared by both `from_pretrained` and `from_backbone`.
Extracted from the script-level `_print_checkpoint_report` we already wrote.

The table structure (same as current scripts):
- Rows per submodule: source = "checkpoint" (green) or "fresh init" (yellow)
- Extra row for shape-skipped count if non-zero (dim, "re-initialized")
- Footer: total params from checkpoint / total fresh init

#### 1c. Override `from_pretrained()` (same-task loading)

```python
@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                    show_load_report=True, **kwargs):
    model, info = super().from_pretrained(
        pretrained_model_name_or_path, *model_args,
        output_loading_info=True, **kwargs,
    )
    if show_load_report:
        # Reconstruct state dict from the loaded file for the report
        state = torch.load(
            os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"),
            map_location="cpu",
        )
        cls._load_report(state, info["missing_keys"],
                         info["unexpected_keys"], shape_skipped=[], model=model)
    return model
```

Note: `output_loading_info=True` is a built-in HF flag that returns missing/unexpected
keys without breaking the normal flow.

#### 1d. Add `from_backbone()` classmethod (cross-task loading)

```python
@classmethod
def from_backbone(cls, backbone_path_or_dir, config, device=None):
    """Load encoder weights from backbone.bin; head is randomly initialized.

    Args:
        backbone_path_or_dir: Path to backbone.bin directly, or a directory
            containing backbone.bin, or a HuggingFace hub repo ID.
        config: CageFusionConfig for the new task (num_labels, model_task, etc.)
        device: torch.device or None (defaults to CUDA if available)

    Returns:
        Model instance with encoder loaded from checkpoint and head fresh-initialized.
    """
```

Internal steps:
1. Resolve the path: if it's a directory, append `/backbone.bin`; if it's a HF hub ID,
   use `hf_hub_download(repo_id, "backbone.bin")`.
2. `state = torch.load(path, map_location="cpu")`
3. Build `model = AutoCageFusion.from_config(config)` (or `cls(config)`)
4. Filter shape-mismatched keys:
   ```python
   model_sd = model.state_dict()
   shape_skipped = [k for k, v in state.items()
                    if k in model_sd and v.shape != model_sd[k].shape]
   compatible = {k: v for k, v in state.items() if k not in shape_skipped}
   missing, unexpected = model.load_state_dict(compatible, strict=False)
   ```
5. Call `cls._load_report(compatible, missing, unexpected, shape_skipped, model)`
6. Print warning:
   ```
   WARNING  Backbone loaded — task head (N params) is randomly initialized.
            Run at least a few training epochs before inference.
   ```
7. Return model (optionally moved to device).

---

### 2. `cage_fusion/auto.py`

Expose `from_backbone` at the `AutoCageFusion` level:

```python
class AutoCageFusion:
    @classmethod
    def from_backbone(cls, backbone_path_or_dir, config, device=None):
        """Cross-task transfer: load encoder, fresh task head."""
        task_cls = _TASK_MAP[config.model_task]
        return task_cls.from_backbone(backbone_path_or_dir, config, device=device)
```

This mirrors the existing `from_config` dispatch pattern.

---

### 3. `cage_fusion/modeling/modeling_cage.py` — `save_pretrained()` override

Override `save_pretrained` to also call `save_backbone()` automatically, so both
files are always written together:

```python
def save_pretrained(self, save_directory, **kwargs):
    super().save_pretrained(save_directory, **kwargs)
    self.save_backbone(save_directory)  # writes backbone.bin alongside pytorch_model.bin
```

This means every `trainer.train()` checkpoint also includes a fresh `backbone.bin`.
No separate call needed.

---

### 4. `cage_fusion/inference/pipeline.py` — `push_to_hub()`

`push_to_hub` currently uploads `pytorch_model.bin` and `config.json`. After the
`save_pretrained` override above, `backbone.bin` will already be in the checkpoint
directory, so it will be included automatically when the HF uploader reads the dir.

No code change needed here — it falls out of step 3.

If it does not pick it up automatically, explicitly add:
```python
api.upload_file(
    path_or_fileobj=os.path.join(checkpoint_dir, "backbone.bin"),
    path_in_repo="backbone.bin",
    repo_id=repo_id,
)
```

---

### 5. Scripts: `pretrain_admet_regression.py` and `pretrain_admet_classification.py`

Replace the manual loading block in both scripts:

**Before:**
```python
state = torch.load(args.init_from_backbone, map_location="cpu")
model_sd = model.state_dict()
shape_skipped = [...]
compatible = {k: v for k, v in state.items() if k not in shape_skipped}
missing, unexpected = model.load_state_dict(compatible, strict=False)
_print_checkpoint_report(...)
```

**After:**
```python
model = AutoCageFusion.from_backbone(args.init_from_backbone, config, device=device)
```

Also remove the `_SUBMODULE_LABELS`, `_SCALE_GROUP`, and `_print_checkpoint_report`
definitions from both scripts — they move into `modeling_cage.py`.

---

### 6. Notebooks (optional, lower priority)

Update `notebooks/03_train_custom_data.ipynb` and
`notebooks/04_train_regression_admet.ipynb` to show both workflows:

```python
# Same-task fine-tuning (full model transfer):
model = AutoCageFusion.from_pretrained("cage-fusion/cage-fusion-pretrained",
                                        config=config)

# Cross-task transfer (encoder only):
model = AutoCageFusion.from_backbone("cage-fusion/cage-fusion-pretrained",
                                      config=new_config)
```

---

## HuggingFace Repo Layout

After push, the single repo `cage-fusion/cage-fusion-pretrained` contains:

```
cage-fusion/cage-fusion-pretrained/
  pytorch_model.bin   ← full model (encoder + regression head, 14 outputs)
  backbone.bin        ← encoder only (safe for any task / any num_labels)
  config.json         ← architecture config (num_labels reflects training task)
  training_args.json  ← hyperparameters
  README.md           ← usage examples for both load paths
```

Users choose:
- `from_pretrained(repo_id)` for continuing regression pretraining
- `from_backbone(repo_id, config=my_config)` for any new task

---

## Implementation Order

1. `save_backbone()` fix (1a) — unblocks everything downstream
2. `_load_report()` helper (1b)
3. `from_backbone()` (1d) — the core new API
4. `save_pretrained()` override (3) — so backbone.bin is always co-saved
5. `from_pretrained()` override (1c) — box on same-task load
6. `AutoCageFusion.from_backbone` dispatch (2)
7. Script cleanup (5) — remove duplicated report code, use new API
8. Notebooks (6) — if time permits

## Open Questions (resolved)

- HF repo: single repo with multiple files (decided)
- Box on from_pretrained: yes, always (decided)
- backbone.bin is now truly task-agnostic after the _HEAD_PREFIXES fix (step 1a)
