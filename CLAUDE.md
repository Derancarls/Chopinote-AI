# CLAUDE.md — Chopinote-AI v0.3.0

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current State

- **Version**: v0.3.0 (tagged, not trained)
- **Branch**: `v0.3.X` (active), `main` (merged)
- **v0.2.x**: abandoned at step ~51000/166000
- **v0.3.0 training**: not started — data migration + training pending
- **Next**: v0.3.1 framework-content separation

## Commands

```bash
# install
pip install -e .

# run tests
python -m pytest tests/ -v

# data migration (v0.2.x → v0.3.0 token IDs)
python scripts/migrate_to_v4.py --input-dir tokens_v3 --output-dir tokens_v4 --num-workers 16

# SSF annotation (generate .ssf.json sidecars)
python scripts/generate_ssf.py annotate --input-dir tokens_v4 --num-workers 16

# figuration annotation (generate .fig.json sidecars)
python scripts/generate_fig.py annotate --input-dir tokens_v4 --num-workers 8

# structure annotation
python scripts/structure_annotator.py annotate \
    --input-dir /root/autodl-tmp/data/processed/tokens_v4 \
    --output-dir /root/autodl-tmp/data/processed/tokens_v4 \
    --num-workers 8

# training (not yet run for v0.3.0)
python scripts/train/run_curriculum_training.py \
    --resume /root/autodl-tmp/chopinote/checkpoints/step_50000.pt \
    --batch-size 8 --phase1-steps 80000 --phase2-steps 40000
```

## Architecture

### v0.3.0 Core Design

**SSF (Sliding Scale Field)**: 12-dim tonic-anchored chroma field replaces discrete key/chord tokens.
- Position 0 = tonic, position i = i semitones from tonic (equivalence classes mod 12)
- TonicField (section-level) + LocalField (bar-level sparse delta)
- Injected via `ssf_proj: nn.Linear(12, d_model)` into hidden state
- `SSFReconstructionHead`: d_model → 12 MSE regression (auxiliary task)
- chord_bias (γ/ε/ζ) removed — SSF field carries chroma information

**Four-Voice Time Slicing**: SATB via `<Voice 0>~<Voice 3>` tokens.
- 43 Program × 4 subtracks (172) + 4 Voice tokens (down from 512 Program tokens)
- `voice_embedding: nn.Embedding(5, d_model)` — zero-init per-voice identity
- `voice_same/samepos` bias (2 learnable scalars) — same-voice history + same-position cross-voice

**Figuration (11 types)**: `fig_embedding: nn.Embedding(12, d_model)` — zero-init

**Cadence (5 types)**: `cadence_embedding: nn.Embedding(6, d_model)` — zero-init

**Vocabulary**: 929 → 542 tokens
- Removed: 30 Key, 30 Anticipate, 21 Chord func/7th/Inv, 508 unused Program
- Added: 12 Tonic, 4 Voice, 12 Fig, 5 Cadence

### Six-Layer ABC Engine v2

```
A1 (框架记忆) → A2 (动机提取) → A3 (统计画像) → B1 (硬约束) → B2 (决策调参) → C (评价层)
```

- **A1** — 段落规划、和声进行 (→ SSF via `harmony_to_ssf()`)、框架回退
- **A2** — 种子分析、动机 DNA 提取、地标管理
- **A3** — 基线统计、逐 bar 密度/rest_ratio、段快照、趋势检测
- **B1** — 上下文禁令、声部音域 (VOICE_RANGE)、平行禁止
- **B2** — 温区退火、创新预算、致命信号检测、参数调节
- **C** — MusicXML 审查、Token↔XML 对比、C→B 反馈

### Packages

- **`chopinote_model/`** — Decoder-only Transformer:
  - `model.py` — MusicTransformer, 24 layers, 1.21B params, RoPE, **QK-Norm**, per-head scaling, **SSF field injection** (ssf_proj), **voice_embedding** (per-voice identity), **voice_bias** (same-voice/same-pos), **fig_embedding**, **cadence_embedding**, sec_bias (α/β/γ/δ), voice_count_embedding, measure_in_section_embedding, SectionPredictionHead (key: 12-dim regression, type: 23-class), SSFReconstructionHead, weight tying, FP8 Linear, logit soft-capping.
    - **Removed** (v0.3.0): chord_embedding, chord_bias (γ/ε/ζ), ChordPredictionHead, `_compute_chord_bias`
  - `config.py` — ModelConfig (vocab_size=542, use_ssf, ssf_dim=12, n_voice_ids=5, n_fig_types=12, n_cadence_types=6, use_chord_attention=False), TrainingConfig (FP8, Z-loss, EMA, dropout schedule, SSF loss weight)
  - `train.py` — Trainer: next_token + sec_type(CE) + key_head(MSE) + SSF_recon(MSE) + Z-loss. No chord loss.
  - `dataset.py` — TokenDataset, auto-loads `.sec.json` + `.ssf.json`, collate_fn pads ssf_fields
  - `generate.py` — **UNCHANGED from v0.2.x** (待 v0.3.1 框架-内容分离重写). Contains old refs to `tokenizer.KEY`/`tokenizer.CHORD_FUNCTIONS` — these functions are not called by current training path.

- **`chopinote_dataset/`** — Data:
  - `tokenizer.py` — REMITokenizer, **vocab=542** (grid_size=16, velocity_levels=8). TONIC_NAMES(12), VOICE_NAMES(4), PROGRAM_NAMES(43), FIGURATION_NAMES(12), CADENCE_NAMES(6). `framework_token_ids` property. `get_tonic_id`/`get_voice_id` helpers.
  - `converter.py` — MusicXML/PDMX/MIDI→REMI. Emits `<Tonic X>` (not `<Key X>`), `<Program N> <Voice M>` (not `<Program N_M>`), no Anticipate tokens.
  - `renderer.py` — REMI→MusicXML (fast path via ElementTree, 54000x speedup)

- **`chopinote_abc/`** — ABC Engine:
  - `planner.py` — plan_structure, plan_harmony, **harmony_to_ssf()** (ChordAtBar→12-dim SSF), chord_func_to_ssf()
  - Others unchanged from v0.2.6

- **`scripts/`** — New for v0.3.0:
  - `migrate_to_v4.py` — v0.2.x→v0.3.0 token ID remapping (Key→Tonic, Program→Program+Voice, with flat-key mapping)
  - `generate_ssf.py` — SSF .ssf.json sidecar annotation (multiprocessing)
  - `generate_fig.py` — Figuration .fig.json sidecar annotation (multiprocessing)

### Key Constraints

- Vocab: 542 tokens, deterministic given grid_size=16, velocity_levels=8
- CloudTrain branch: local/remote diverged, write-only push only
- Git commits: `Derancarls <derancarls@foxmail.com>`, **NO** `Co-Authored-By`
- `docs/` directory: NEVER delete from disk
- `chopinote_evaluator/`: deleted, logic absorbed into `chopinote_abc/`
- DataLoader: `num_workers=0` mandatory (multiprocessing crashes on long runs)
- bf16 training: no GradScaler needed
- generate.py: will be rewritten in v0.3.1 (framework-content separation)

### Design Documents (v0.3.x)

| Document | Topic |
|----------|-------|
| `docs/ssf_encoding_v0.3.x.md` | SSF tonic-anchored chroma field |
| `docs/voice_time_slicing_v0.3.x.md` | Four-voice time slicing + Voice bias |
| `docs/figuration_encoding_v0.3.x.md` | Piano figuration types (11 classes) |
| `docs/framework_content_separation_v0.3.x.md` | Framework/content separation (v0.3.1) |
| `docs/cadence_awareness_v0.3.x.md` | Cadence zone + embedding |
| `docs/phrase_layer_design_v0.3.x.md` | Phrase layer design + v0.3.x adaptations |
| `docs/duration_saturation_v0.3.x.md` | Duration saturation — 17 buckets, B1 guard |
| `docs/voice_splitting_v0.3.x.md` | Piano 2-track → 4-voice split + VoicePlan (seed detect + user override) |
| `docs/curriculum_training_v0.3.x.md` | F1-F5 quality filter + 4-metric classification + 3-phase curriculum |
| `docs/ssf_pairwise_bias_backup.md` | SSF pairwise similarity bias (backup) |
| `docs/remi_z_track_continuous.md` | REMI-z track-continuous interleaving (backup) |
| `docs/v0.2.6_audit_v0.3.x_gap.md` | v0.2.6 pain points × v0.3.x coverage |
