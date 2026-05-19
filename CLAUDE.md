# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# install
pip install -e .

# run all tests
python -m pytest tests/ -v

# run single test file
python -m pytest tests/test_tokenizer.py -v

# run single test
python -m pytest tests/test_tokenizer.py::TestVocabSize -v

# CLI inference
chopin checkpoints/step_X.pt input.musicxml

# launch control (点火控制台)
python scripts/train/launch_control.py check          # 预检清单
python scripts/train/launch_control.py launch         # 全流程点火 + 监控
python scripts/train/launch_control.py monitor        # 实时仪表盘
python scripts/train/launch_control.py abort          # 中止一切
python scripts/train/launch_control.py status         # 状态快照

# data preprocessing (MIDI → REMI tokens)
python scripts/preprocess/run_fast_preprocess.py

# curriculum training (manual)
python scripts/train/run_curriculum_training.py \
    --midi-train-list /root/autodl-tmp/data/processed/train.txt \
    --musicxml-train-list /root/autodl-tmp/data/processed/train.txt \
    --val-list /root/autodl-tmp/data/processed/val.txt \
    --data-dir /root/autodl-tmp/data/processed \
    --phase1-steps 120000 --phase2-steps 50000 \
    --batch-size 8 \
    --output-dir /root/autodl-tmp/chopinote/checkpoints \
    --log-dir /root/autodl-tmp/chopinote/tensorboard \
    --resume /root/autodl-tmp/chopinote/checkpoints/step_X.pt
```

## Architecture

### Packages

- **`chopinote_model/`** — Decoder-only Transformer (MusicTransformer):
  - `model.py` — MusicTransformer (GPT decoder), CausalSelfAttention (nn.Embedding rel/measure bias, SDPA sdpa_kernel, causal fused into attn_mask, weight tying, use_reentrant=True checkpointing)
  - `config.py` — ModelConfig (vocab_size=872, d_model=2048, n_layers=24, n_heads=32, d_ff=8192, ~1.21B params), TrainingConfig, PhaseConfig, TokenLossMask
  - `train.py` — Trainer class, single-phase + multi-phase curriculum training, AMP bf16 (autocast + direct backward, no GradScaler), AdamW, cosine LR, TensorBoard
  - `dataset.py` — TokenDataset (token_lengths.json index, LRU cache 128), collate_fn (dynamic padding)
  - `generate.py` — autoregressive generation (KV cache, top-k, pitch constraints), MusicXML export

- **`chopinote_dataset/`** — Data processing:
  - `tokenizer.py` — REMITokenizer (fixed vocab 872, grid_size=16, velocity_levels=8), build_vocab() is deterministic
  - `converter.py` — MusicXMLToREMI, PDMXToREMI, MIDIToREMI
  - `fast_converter.py` — FastMIDIToREMI (mido-based, ~80x faster)
  - `processor.py` — batch preprocessor base classes
  - `splitter.py` — dataset train/val/test split

- **`chopinote_cli/`** — CLI entry (`chopin` command):
  - `main.py` — inference CLI with preset support
  - `presets.py` — generation style presets (baroque, romantic, jazz, etc.)

- **`scripts/`** — Utility scripts, organized by function:
  - `preprocess/` — data preprocessing (MIDI/PDMX/MusicXML → REMI): `run_fast_preprocess.py`, `rerun_musicxml.py`, `rerun_pdmx.py`, `rerun_all_v3.py`, `dedup_and_split.py`, `generate_token_lengths.py`
  - `train/` — training: `run_curriculum_training.py` (two-phase, argparse, TF32), `dpo_train.py` (DPO fine-tune), `launch_control.py` (点火控制台, preflight → launch → monitor → abort)
  - `generate/` — generation & testing: `batch_roundtrip.py`, `roundtrip_test.py`, `validate_generation.py`, seed utilities
  - `analysis/` — analysis & verification: `align_converter.py`, `verify_e2e.py`, `analyze_tokens.py`, `analyze_notations.py`

### Hardware & Training Setup

- **GPU**: RTX 5090 32GB, bf16 training
- **Batch**: batch_size=8, grad_accum=4 (effective batch=32)
- **Data**: `/root/autodl-tmp/data/processed/` (~1.6M token files, 13.7B tokens, 400G disk)
- **Checkpoints**: `/root/autodl-tmp/chopinote/checkpoints/`
- **Logs/TensorBoard**: `/root/autodl-tmp/chopinote/tensorboard/` (also symlinked at `/root/tf-logs`)
- **Memory-critical**: attention bias bf16 (nn.Embedding), padding mask avoids (B,nH,T,T) broadcast, measure_bias shares first sample, SDPA sdpa_kernel restricts to flash/mem_efficient

### Memory Optimization (RTX 5090)

- `torch.set_float32_matmul_precision('high')` — TF32 matmul
- `torch.backends.cudnn.benchmark = True`
- SDPA via `sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION])` context manager — restricts to flash/mem_efficient, prevents math backend OOM
- Causal mask fused into `attn_mask` (not `is_causal=True`) so flash attention works with non-null mask
- `nn.Embedding` for rel/measure bias (bf16 params, simpler than manual indexing)
- Padding mask uses `mask[0]` only (all samples in batch have same padding pattern)
- Measure bias uses first sample's measure structure → (1, nH, T, T) not (B, nH, T, T)
- bf16 autocast + direct `loss.backward()` (no GradScaler needed)
- Gradient checkpointing on full TransformerBlock (attn+FFN), `use_reentrant=True`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Key Constraints

- Vocab size and token IDs are **fixed** given same grid_size/velocity_levels — tokenizer is deterministic
- The local `CloudTrain` branch and `origin/CloudTrain` are intentionally diverged — never fetch/pull/merge from origin (write-only push only)
- Padding across batch is always identical (same file list), so single-sample padding mask suffices
- **`docs/` directory is NEVER to be deleted from disk** — it contains local design notes, gitignored but must stay on filesystem
