# Codebase Consistency Audit

Date: 2026-05-16

## Critical Bugs

### C1. `measure_ids` passed to `model.forward()` but not accepted

**Location:**
- `chopinote_cli/main.py:417` — passes `measure_ids=cached_measure_ids` 
- `chopinote_model/model.py:196` — `forward()` signature has no `measure_ids` param

**Impact:** `generate_with_progress()` crashes at runtime with `TypeError`.
The model computes `measure_ids` internally from `input_ids`, but with KV cache only
1 token is fed per step, making the internal computation wrong (cumsum of a single
token gives 0 or 1 instead of the global measure count).

**Fix needed:** Add optional `measure_ids` param to `MusicTransformer.forward()`.
When provided, use it directly. When absent, compute from `input_ids` as before.

The same issue exists in `chopinote_model/generate.py:295` (simple `generate()`
function) — the `generate` function also calls `model.forward(next_token, kv_caches=kv_caches)`
with single-token inputs, leading to incorrect measure embeddings during generation.

### C2. Optimizer state discarded on checkpoint resume

**Location:** `chopinote_model/train.py`

**Flow:**
1. `load_checkpoint()` loads optimizer state into `self.optimizer` (created in `__init__`)
2. `_run_training_loop()` creates a **new** `AdamW` optimizer, overwriting `self.optimizer`
3. The loaded optimizer state is lost → resume training starts with fresh momentum

**Impact:** When resuming from a checkpoint mid-training, the Adam optimizer momentum
and variance estimates are reset. Training continues but takes extra steps to recover.

## Warnings

### W1. Duplicate `_key_name_to_tonic_midi` in fast_converter.py

`chopinote_dataset/fast_converter.py:54` defined its own `_key_name_to_tonic_midi`
instead of importing from `chopinote_dataset.tokenizer.key_name_to_tonic_midi`.
The duplicate was identical.

**Status: FIXED** — Replaced with import from `tokenizer` module.

### W2. `run_curriculum_training.py` passes `val_dataloader=None`

`scripts/run_curriculum_training.py:141` — `trainer.train(val_dataloader=None)`.
The per-type accuracy metrics wouldn't be computed during training because no
validation dataloader was provided.

**Status: FIXED** — When `--val-list` is provided, a validation DataLoader is now created and passed to `trainer.train()`.

### W3. `generate.py`'s `generate()` missing attention_mask passthrough

`chopinote_model/generate.py:295` — OK for single-sequence inference
(padding not needed, `is_causal=True` handles masking), but inconsistent
with the training path.

## Consistency Check Summary

| Interface | Status |
|-----------|--------|
| ModelConfig → MusicTransformer | ✓ |
| MusicTransformer.forward → CausalSelfAttention | ✓ |
| Trainer → TokenDataset | ✓ |
| Trainer → collate_fn | ✓ |
| TokenLossMask → REMITokenizer | ✓ |
| `chopinote_cli/main.py` → `chopinote_model.generate` | ✓ (imports) |
| `chopinote_cli/main.py` → `MusicTransformer.forward()` | **C1 — measure_ids bug** |
| Checkpoint load → optimizer state persistence | **C2 — state lost** |
| `chopinote_model/__init__.py` exports | ✓ (Trainer/PhaseConfig used via direct import) |
| Tokenizer constants → converter constants | ✓ (grid_size=16, velocity_levels=8) |
