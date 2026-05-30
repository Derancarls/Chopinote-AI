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
chopin checkpoints/step_47000.pt input.musicxml

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

# ABC Engine full cycle generation test
python scripts/generate/test_abc_full_cycle.py

# batch evaluation (fill reward_log for DPO)
python scripts/train/batch_evaluate.py \
    --checkpoint /root/autodl-tmp/chopinote/checkpoints/step_50000.pt \
    --seeds /root/Chopinote-AI/data/eval_seeds.txt \
    --temperatures 0.9,1.1 --samples-per-seed 1 --max-bars 32

# DPO training (standalone)
python scripts/train/dpo_train.py \
    --checkpoint /root/autodl-tmp/chopinote/checkpoints/step_50000.pt \
    --reward-dir /root/autodl-tmp/chopinote/rewards \
    --epochs 3

# curriculum training with DPO auto-loop
python scripts/train/run_curriculum_training.py \
    --resume /root/autodl-tmp/chopinote/checkpoints/step_50000.pt \
    --dpo-enabled --dpo-interval 5000 --dpo-min-entries 20 \
    --eval-enabled --eval-interval 5000 \
    --eval-seeds /root/Chopinote-AI/data/eval_seeds.txt \
    --phase1-steps 120000 --phase2-steps 50000 \
    --batch-size 8 \
    --output-dir /root/autodl-tmp/chopinote/checkpoints \
    --log-dir /root/autodl-tmp/chopinote/tensorboard
```

## Architecture

### Six-Layer ABC Engine v2

```
A1 (框架记忆) → A2 (动机提取) → A3 (统计画像) → B1 (硬约束) → B2 (决策调参) → C (评价层)
```

- **A1** — 段落规划、和声进行、框架回退
- **A2** — 种子分析、动机 DNA 提取（contour/rhythm/scale_degrees/ambitus）、地标管理
- **A3** — 基线统计、逐 bar 密度/rest_ratio/b1_score、段快照、趋势检测
- **B1** — 上下文禁令（Program/Tempo/TimeSig/Tuplet/GraceNote）、逐 bar 动态禁令、声部音域/平行禁止、硬约束通过状态
- **B2** — 温区退火（冷→热→冷）、创新预算管理、致命信号检测（reharmonize/abort）、参数调节
- **C** — 段对比（PC_sim/density_sim）、MusicXML 快速审查、Token↔XML 保真度对比、C→B 结构化反馈、最终评分

### Packages

- **`chopinote_model/`** — Decoder-only Transformer:
  - `model.py` — MusicTransformer, 24 layers, RoPE positional encoding, **QK-Norm** (per-head RMSNorm), **per-head Q/K scaling**, section-aware attention (sec_bias: α/β/γ/δ learnable + bar distance decay), chord-aware attention (chord_bias: γ/ε/ζ + δ sec_bias dedup), **voice_count_embedding** (同位置声部计数), **measure_in_section_embedding** (节内相对位置), SectionPredictionHead (key/type), ChordPredictionHead (func/inv), weight tying, gradient checkpointing, FP8 Linear, **attention logit soft-capping** (manual fallback)
  - `config.py` — ModelConfig (vocab_size=929, d_model=2048, n_layers=24, n_heads=32, d_ff=8192, ~1.21B params), TrainingConfig (FP8 enabled by default, **Z-loss**, **EMA β=0.999**, **dropout schedule** 0.15→0.10→0.08, token-level weighted CE loss, **DPO 自动微调配置**, **自动评估生成配置**), PhaseConfig, TokenLossMask
  - `train.py` — Trainer class, multi-task loss (next_token + sec_pred + chord_pred + **Z-loss**), **EMA weight tracking** (saved in ckpt), single/multi-phase curriculum, AMP bf16, AdamW, cosine LR, TensorBoard, per-token-type accuracy evaluation, **dropout step-based scheduling**, **DPO auto-trigger**: `_check_dpo_trigger()` → `_run_dpo_phase()`, **eval auto-trigger**: `_run_eval_generation()` → fills reward_log
  - `dataset.py` — TokenDataset (token_lengths.json index, LRU cache 128), auto-loads `.sec.json` and `.chord.json` sidecars, **voice_count_ids** + **measure_in_section_ids** per-token, collate_fn (dynamic padding for all fields)
  - `generate.py` — **ABC Engine full pipeline**: `stage3_iterative_generate()` — Stage 1 (structure plan) → Stage 2 (harmony skeleton) → Stage 3 (note filling with sec_bias+chord_bias, B1 hard bans, B2 zone temperature), `_c_evaluate()` with MusicXML review + Token↔XML comparison + C→B feedback, per-section `_apply_section_c_feedback()`, KV cache, context token bans (Program seed-preserving)

- **`chopinote_dataset/`** — Data processing:
  - `tokenizer.py` — REMITokenizer (fixed vocab 929, grid_size=16, velocity_levels=8), 16 chord functions + Chord7 + 4 inversions, build_vocab() is deterministic
  - `converter.py` — MusicXMLToREMI, PDMXToREMI, MIDIToREMI
  - `fast_converter.py` — FastMIDIToREMI (mido-based, ~80x faster)
  - `processor.py` — batch preprocessor base classes
  - `splitter.py` — dataset train/val/test split

- **`chopinote_abc/`** — ABC Engine v2 (六层架构):
  - `database.py` — A1DB (框架记忆库: SectionPlan, ChordAtBar, SeedContext, and声回退), A2DB (动机摘要库: MotifDNA, MotifRecord, 地标管理), A3DB (统计画像库: BarStats, 基线/baseline/趋势, write_reward_log with seed_bars+total_score), StructuralFix dataclass
  - `planner.py` — Stage 1/2 规则规划器 (plan_structure: sonata/binary/theme_variations/free 曲式, plan_harmony: 功能和声模板+终止式, reharmonize_from_bar: 和声回退 with seed_bar_offset)
  - `motif.py` — A2 动机提取 (identify_landmarks, purify_tokens, extract_dna: contour/rhythm/scale_degrees/ambitus, contour_similarity, invert_contour)
  - `decision.py` — B1 (BHardBans: 上下文禁令+声部音域+平行禁止) + B2 (BFeedback: 致命信号/创新预算/发展配方, apply_zone_temperature: 冷→热→冷曲线, InnovationEntry)
  - `metrics.py` — 27+ token 级指标 (pitch_class_kl, interval_kl, key_consistency, max_polyphony_check, voice_crossing, compute_all_metrics 等)
  - `constraints.py` — Token 级 + Score 级规则检查 (parallel_fifths, voice_crossing_ tokens, out_of_range, harmony_progression, evaluate_theory)
  - `scoring.py` — C 评价层: EvalReport, evaluate_generation (合法性+理论+一致性+连贯性), **MusicXML 快速审查** (BarInspection: 逐 bar 声部沉默/音域违规/声部交错/平行五八度), **Token↔XML 对比** (compare_tokens_to_xml: roundtrip 保真度), **C→B 结构化反馈** (CFeedback: ban_pitches/part_bias/temperature_delta/complexity_delta/fatal, c_review_to_feedback, apply_c_feedback_to_bans)
  - `parser.py` — MusicXML → Score 中间表示 (Note/Measure/Score dataclasses, parse_musicxml via music21)
  - `logging.py` — ABCGenerationLogger: 每次生成大循环一个独立日志文件 (logs/generate/), 六层全覆盖 (A1/A2/A3/B1/B2/C), 双输出 (DEBUG 全量文件 + WARNING+ 控制台), JSON 汇总 (.summary.json)
  - `__init__.py` — 统一导出

- **`chopinote_cli/`** — CLI entry (`chopin` command):
  - `main.py` — inference CLI with preset + ABC Engine integration (calls stage3_iterative_generate), save_to_musicxml helper
  - `presets.py` — generation style presets (baroque, romantic, jazz, etc.)

- **`scripts/`** — Utility scripts, organized by function:
  - `preprocess/` — data preprocessing (MIDI/PDMX/MusicXML → REMI): `run_fast_preprocess.py`, `rerun_musicxml.py`, `rerun_pdmx.py`, `rerun_all_v3.py`, `dedup_and_split.py`, `generate_token_lengths.py`
  - `train/` — training: `run_curriculum_training.py` (two-phase, argparse, TF32, DPO CLI args, eval CLI args), `dpo_train.py` (DPO standalone: build_preference_dataset, LoRALinear, dpo_loss, compute_log_probs, DPODataLoader), `batch_evaluate.py` (批量生成+评价: run_batch_evaluation for train.py inline or standalone CLI), `launch_control.py` (点火控制台, preflight → launch → monitor → abort)
  - `generate/` — generation & testing: `test_abc_full_cycle.py` (sonata 48bar with ABCGenerationLogger), `batch_roundtrip.py`, `roundtrip_test.py`, `validate_generation.py`
  - `analysis/` — analysis & verification: `align_converter.py`, `verify_e2e.py`, `analyze_tokens.py`, `analyze_notations.py`
  - `structure_annotator.py` — section boundary detection + type inference (Sonata/Theme/Fallback)
  - `chord_annotator.py` — chord function annotation via template matching + key context + confidence > 0.8

### DPO Auto-Optimization Loop

```
训练 step N → save_checkpoint
  → _run_eval_generation(): 批量生成 (seed×temp×samples) → C 评分 → write_reward_log
  → _check_dpo_trigger(): reward_log 新增 ≥ dpo_min_new_entries?
    → 否 → 跳过
    → 是 → build_preference_dataset() → LoRA(QKV only, rank=8) → DPO train 3 epoch → merge → continue
```

All eval/DPO flags default **off**. Enable with `--eval-enabled --dpo-enabled`.

### Hardware & Training Setup

- **GPU**: RTX 4090 48GB, bf16 training
- **Batch**: batch_size=8, grad_accum=2 (effective batch=16)
- **Data**: `/root/autodl-tmp/data/processed/` (~1.37M token files, 13.7B tokens, 400G disk)
- **Checkpoints**: `/root/autodl-tmp/chopinote/checkpoints/`
- **Reward log**: `/root/autodl-tmp/chopinote/rewards/reward_log.jsonl`
- **Logs/TensorBoard**: `/root/autodl-tmp/chopinote/tensorboard/` (also symlinked at `/root/tf-logs`)
- **ABC Engine logs**: `/root/Chopinote-AI/logs/generate/` (per-session .log + .summary.json)
- **Memory-critical**: manual attention path when sec_bias present (avoids SDPA), padding mask shares first sample, SDPA sdpa_kernel restricts to flash/mem_efficient

### Memory Optimization (RTX 4090 48GB)

- `torch.set_float32_matmul_precision('high')` — TF32 matmul
- `torch.backends.cudnn.benchmark = True`
- SDPA via `sdpa_kernel([CUDNN, FLASH_ATTENTION, EFFICIENT_ATTENTION])` context manager
- Causal mask fused into `attn_mask` (not `is_causal=True`) so flash attention works with non-null mask
- Padding mask uses `mask[0]` only (all samples in batch have same padding pattern)
- Measure embedding uses first sample's measure structure
- **sec_bias/chord_bias: recomputed inside checkpoint from raw (B,T) data (~1 MiB) instead of storing (B,1,T,T)=256 MiB → 24 layers save 6 GiB. Combined bias detached before SDPA to avoid 8 GiB attention matrix materialization during backward (PyTorch built-in efficient attention materializes full (B,nH,T,T) when computing dL/d(attn_mask)).**
- bf16 autocast + direct `loss.backward()` (no GradScaler needed)
- Gradient checkpointing on full TransformerBlock (attn+FFN), `use_reentrant=True`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `weakref.ref(self)` per block to enable bias recomputation without nn.Module registration cycle

### Key Constraints

- Vocab size and token IDs are **fixed** given same grid_size/velocity_levels — tokenizer is deterministic
- The local `CloudTrain` branch and `origin/CloudTrain` are intentionally diverged — never fetch/pull/merge from origin (write-only push only)
- Padding across batch is always identical (same file list), so single-sample padding mask suffices
- **`docs/` directory is NEVER to be deleted from disk** — it contains local design notes, gitignored but must stay on filesystem
- **Git commits** — 必须用 `Derancarls <derancarls@foxmail.com>` 名义，**禁止**加 `Co-Authored-By` 行，所有代码贡献只以用户名义记录
- **`chopinote_evaluator/` has been deleted** — all evaluation logic absorbed into `chopinote_abc/`. Do NOT recreate it.
