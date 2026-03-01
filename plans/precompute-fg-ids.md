# Plan: Precompute Functional Group IDs at Featurization Time

## Problem

`FunctionalGroupPrompt.forward()` in `cage_fusion/modeling/modules/fg_prompt.py` calls
`Chem.MolFromSmiles(smiles)` on every forward pass (train + eval) to detect functional groups.

This is wrong for two reasons:
1. **Redundant work** — molecules were already fully parsed during featurization. The mol
   objects are available at that point but are discarded after graph bytes and SMILES are stored.
2. **RDKit warnings** — some SMILES trigger "not removing hydrogen atom without neighbors"
   warnings during sanitization. Currently suppressed with a `_mol_from_smiles_quiet` helper,
   but the parse itself is still happening on every step.

## Root Cause

The pipeline stores canonical SMILES strings in the HDF5 file and passes them through the
DataLoader to the model. `FunctionalGroupPrompt` then re-parses those strings and runs SMARTS
matching. FG IDs (the output of this process) should instead be precomputed once and cached.

## Proposed Fix: Precompute FG IDs at Featurization Time

Store FG IDs as a new `fg_ids` VarLen int16 dataset in the HDF5 file. The dataset, collator,
and model forward pass all get updated to carry these through.

### Backward Compatibility

- Old HDF5 files (without `fg_ids`) are handled gracefully: `dataset.py` sets
  `self.has_fg_ids = False` and returns an empty list per sample.
- `FunctionalGroupPrompt.forward()` falls back to the current SMILES-based detection path
  when `fg_ids_batch` is `None` or empty.
- After re-featurizing, `_mol_from_smiles_quiet` and the fallback path can be removed.

---

## Files to Change (in order)

### 1. `cage_fusion/featurization/featurizer_utils.py`

**Add** `process_fg_ids()`:
```python
from cage_fusion.chemistry.fg_utils import get_functional_groups

def process_fg_ids(batch_df) -> List[List[int]]:
    """Compute functional group IDs for each molecule in the batch."""
    return [get_functional_groups(row.mol) for row in batch_df.itertuples(index=False)]
```

**Update** `initialize_hdf5_file()`: add `fg_ids_enabled: bool = True` parameter.
When True, create a VarLen int16 dataset:
```python
if fg_ids_enabled:
    vlen_int16 = h5py.vlen_dtype(np.int16)
    f.create_dataset("fg_ids", shape=(num_samples,), maxshape=(None,), dtype=vlen_int16)
```

---

### 2. `cage_fusion/featurization/molecular_featurizer.py`

In `featurize_and_save_streaming()`:
- Import `process_fg_ids` from `featurizer_utils`.
- In the featurization loop, compute and store:
  ```python
  batch_fg_ids = process_fg_ids(batch_df)
  batch_fg_ids_arrays = [np.array(ids, dtype=np.int16) for ids in batch_fg_ids]
  f["fg_ids"][write_idx : write_idx + bs] = batch_fg_ids_arrays
  ```
- Add `"fg_ids"` to the end-of-loop resize list.

---

### 3. `cage_fusion/data/dataset.py`

In `CageFusionStreamingDataset.__init__()`:
- Set `self.has_fg_ids = "fg_ids" in f`.
- Cache fg_ids in RAM (they are tiny — ~few KB for 28k molecules):
  ```python
  if self.has_fg_ids:
      self._fg_ids = [list(f["fg_ids"][i].astype(int)) for i in range(self.length)]
  else:
      self._fg_ids = None
  ```

In `__getitem__()`:
- Fetch `fg_ids = self._fg_ids[idx] if self._fg_ids is not None else []`
- Add `fg_ids` to the return tuple between `smiles` and `original_index`:

**Current return (8-tuple):**
```python
return graph, token_embs, aux, labels, token_ids, smiles, original_index, id_str
```
**New return (9-tuple):**
```python
return graph, token_embs, aux, labels, token_ids, smiles, fg_ids, original_index, id_str
```

---

### 4. `cage_fusion/data/collator.py`

Unpack the 9-tuple and output a 10-tuple:

**Current output (9-tuple):**
```
batched_graph, embeddings, attn_mask, aux_features, labels, input_ids,
smiles, original_indices, ids_list
```
**New output (10-tuple):**
```
batched_graph, embeddings, attn_mask, aux_features, labels, input_ids,
smiles, fg_ids_batch, original_indices, ids_list
```

`fg_ids_batch` is `List[List[int]]` — just `list(fg_ids_per_sample)` from the zip.
No tensor conversion needed; `FunctionalGroupPrompt` consumes it as Python lists.

---

### 5. `cage_fusion/training/trainer.py`

Both `train_epoch()` and `evaluate()` unpack the batch. Update from:
```python
bmg, token_embs, attn_mask, aux_feats, labels, input_ids, smiles_batch, _, _ = batch
```
to:
```python
bmg, token_embs, attn_mask, aux_feats, labels, input_ids, smiles_batch, fg_ids_batch, _, _ = batch
```
Pass `fg_ids_batch=fg_ids_batch` to the model forward call.

---

### 6. `cage_fusion/modeling/modules/fg_prompt.py`

Change `forward()` signature from:
```python
def forward(self, smiles_batch, atom_features, bmg, fg_detector, return_attn=False)
```
to:
```python
def forward(
    self,
    atom_features,
    bmg,
    return_attn=False,
    fg_ids_batch=None,        # precomputed (preferred)
    smiles_batch=None,        # fallback when fg_ids not available
    fg_detector=None,
)
```

In the loop body, replace the `_mol_from_smiles_quiet` + SMARTS matching block with:
```python
if fg_ids_batch is not None:
    fg_ids = fg_ids_batch[i]  # already computed
else:
    # fallback: parse SMILES and detect (backward compat)
    mol = _mol_from_smiles_quiet(smiles_batch[i])
    fg_ids = fg_detector(mol) if mol is not None else []
```

Once all HDF5 caches are regenerated, the fallback branch + `_mol_from_smiles_quiet` can be
removed entirely.

---

### 7. `cage_fusion/modeling/modeling_cage.py`

Add `fg_ids_batch: Optional[List[List[int]]] = None` to `CAGEFusionModel.forward()` (and the
two task-head forward signatures that delegate to it).

Update the FG prompt call condition from:
```python
if cfg.use_fg_prompt and self.fg_prompter is not None and smiles_batch:
```
to:
```python
if cfg.use_fg_prompt and self.fg_prompter is not None and (fg_ids_batch is not None or smiles_batch):
```

Update the `self.fg_prompter(...)` call to pass both:
```python
fg_prompt, prompt_attn_weights = self.fg_prompter(
    atom_features=atom_features,
    bmg=bmg,
    return_attn=return_attn,
    fg_ids_batch=fg_ids_batch,
    smiles_batch=smiles_batch,
    fg_detector=get_functional_groups,
)
```

---

### 8. `cage_fusion/inference/pipeline.py`

Update batch unpack from 9-tuple to 10-tuple:
```python
(bmg, token_embs, attn_mask, aux_feats, labels, input_ids,
 smiles_batch, fg_ids_batch, original_indices_batch, ids_list) = batch
```
Pass `fg_ids_batch=fg_ids_batch` to the model forward call.

---

## After Implementation

- Re-featurize (delete stale `.h5` files) to get `fg_ids` into HDF5.
- `MolFromSmiles` will no longer be called during training/eval/inference.
- The `_mol_from_smiles_quiet` helper and fallback branch in `fg_prompt.py` can be cleaned up.

## Note on `smiles_batch`

`smiles_batch` is still needed in the collator/trainer/pipeline for:
- Logging / provenance
- Attention visualisation (`visualize_fg_attention` in inference pipeline)
- The fallback FG detection path (until re-featurized)

It should not be removed.
