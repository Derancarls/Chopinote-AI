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

# structure annotation (段落标注, 生成 .sec.json)
python scripts/structure_annotator.py annotate \
    --input-dir /root/autodl-tmp/data/processed/tokens_v3 \
    --output-dir /root/autodl-tmp/data/processed/tokens_v3 \
    --num-workers 8

# chord annotation (和弦标注, 生成 .chord.json)
python scripts/chord_annotator.py \
    --tokens-dir /root/autodl-tmp/data/processed/tokens_v3 \
    --file-list /root/autodl-tmp/data/processed/train.txt \
    --output-suffix .chord.json

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

- **`chopinote_model/`** — Decoder-only Transformer:
  - `model.py` — MusicTransformer, 24 layers, RoPE positional encoding, section-aware attention (sec_bias: α/β/γ/δ learnable + bar distance decay), chord-aware attention (chord_bias: γ/ε/ζ + δ sec_bias dedup), SectionPredictionHead (bars/key/type), ChordPredictionHead (func/inv), weight tying, gradient checkpointing, FP8 Linear optional
  - `config.py` — ModelConfig (vocab_size=929, d_model=2048, n_layers=24, n_heads=32, d_ff=8192, ~1.21B params), TrainingConfig, PhaseConfig, TokenLossMask, section/chord config fields
  - `train.py` — Trainer class, multi-task loss (next_token + sec_pred + chord_pred), single/multi-phase curriculum, AMP bf16, AdamW, cosine LR, TensorBoard, per-token-type accuracy evaluation
  - `dataset.py` — TokenDataset (token_lengths.json index, LRU cache 128), auto-loads `.sec.json` and `.chord.json` sidecars, collate_fn (dynamic padding for all fields)
  - `generate.py` — Three-stage generation: Stage 1 (structure plan) → Stage 2 (harmony skeleton, Chord→Inv state machine) → Stage 3 (note filling with sec_bias+chord_bias), KV cache, section-aware context tracking

- **`chopinote_dataset/`** — Data processing:
  - `tokenizer.py` — REMITokenizer (fixed vocab 929, grid_size=16, velocity_levels=8), 16 chord functions + Chord7 + 4 inversions, build_vocab() is deterministic
  - `converter.py` — MusicXMLToREMI, PDMXToREMI, MIDIToREMI
  - `fast_converter.py` — FastMIDIToREMI (mido-based, ~80x faster)
  - `processor.py` — batch preprocessor base classes
  - `splitter.py` — dataset train/val/test split

- **`chopinote_evaluator/`** — Music evaluation:
  - `feedback_controller.py` — A/B1/B2/C feedback loop: PreGenerationEvaluator (seed profile), NarrowFeedbackController (per-bar B1 local + B2 global drift), PostGenerationFilter (C phase scoring + rollback retry + RL reward logging)
  - `registry.py` — Metric registry, token-level implementations for 20+ metrics including 4 chord metrics (chord_melody_alignment, progression_validity, cadence_quality, harmonic_rhythm)
  - `score.py` — Evaluator (MusicXML-level scoring with benchmarks)
  - `report.py` — Evaluation report generation
  - `general/` — General metrics (statistics, legality, theory, harmony)
  - `specific/` — Specific metrics (coherence, consistency, model_scorer)

- **`chopinote_cli/`** — CLI entry (`chopin` command):
  - `main.py` — inference CLI with preset + feedback mode + interactive retry
  - `presets.py` — generation style presets (baroque, romantic, jazz, etc.)

- **`scripts/`** — Utility scripts, organized by function:
  - `preprocess/` — data preprocessing (MIDI/PDMX/MusicXML → REMI): `run_fast_preprocess.py`, `rerun_musicxml.py`, `rerun_pdmx.py`, `rerun_all_v3.py`, `dedup_and_split.py`, `generate_token_lengths.py`
  - `train/` — training: `run_curriculum_training.py` (two-phase, argparse, TF32), `dpo_train.py` (DPO fine-tune), `launch_control.py` (点火控制台, preflight → launch → monitor → abort)
  - `generate/` — generation & testing: `batch_roundtrip.py`, `roundtrip_test.py`, `validate_generation.py`, seed utilities
  - `analysis/` — analysis & verification: `align_converter.py`, `verify_e2e.py`, `analyze_tokens.py`, `analyze_notations.py`
  - `structure_annotator.py` — section boundary detection + type inference (Sonata/Theme/Fallback)
  - `chord_annotator.py` — chord function annotation via template matching + key context + confidence > 0.8

### Hardware & Training Setup

- **GPU**: RTX 5090 32GB, bf16 training
- **Batch**: batch_size=8, grad_accum=4 (effective batch=32)
- **Data**: `/root/autodl-tmp/data/processed/` (~1.37M token files, 13.7B tokens, 400G disk)
- **Checkpoints**: `/root/autodl-tmp/chopinote/checkpoints/`
- **Logs/TensorBoard**: `/root/autodl-tmp/chopinote/tensorboard/` (also symlinked at `/root/tf-logs`)
- **Memory-critical**: manual attention path when sec_bias present (avoids SDPA), padding mask shares first sample, SDPA sdpa_kernel restricts to flash/mem_efficient

### Memory Optimization (RTX 5090)

- `torch.set_float32_matmul_precision('high')` — TF32 matmul
- `torch.backends.cudnn.benchmark = True`
- SDPA via `sdpa_kernel([CUDNN, FLASH_ATTENTION, EFFICIENT_ATTENTION])` context manager — cuDNN attention preferred, fallback to flash/mem_efficient
- Causal mask fused into `attn_mask` (not `is_causal=True`) so flash attention works with non-null mask
- Padding mask uses `mask[0]` only (all samples in batch have same padding pattern)
- Measure embedding uses first sample's measure structure
- sec_bias triggers manual attention path (no SDPA); without sec_bias, standard SDPA fast path
- bf16 autocast + direct `loss.backward()` (no GradScaler needed)
- Gradient checkpointing on full TransformerBlock (attn+FFN), `use_reentrant=True`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Key Constraints

- Vocab size and token IDs are **fixed** given same grid_size/velocity_levels — tokenizer is deterministic
- The local `CloudTrain` branch and `origin/CloudTrain` are intentionally diverged — never fetch/pull/merge from origin (write-only push only)
- Padding across batch is always identical (same file list), so single-sample padding mask suffices
- **`docs/` directory is NEVER to be deleted from disk** — it contains local design notes, gitignored but must stay on filesystem
