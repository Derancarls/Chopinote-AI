# CLAUDE.md — Chopinote-AI v0.3.3

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current State

- **Version**: v0.3.3 (code-complete, not trained)
- **Branch**: `v0.3.X` (active), `main` (merged)
- **v0.2.x**: abandoned at step ~51000/166000
- **v0.3.3 annotation**: SSF / Fig / Func v3 三管线完成, 350GB LMDB, 1.6M files
- **v0.3.3 opt1~opt5**: 五作曲能力模块全部代码完成 ✅
- **Next**: v0.3.4 — VoicePlan generation-side, Cadence Zone generation-side, hierarchical generation, training

## Commands

```bash
# install
pip install -e .

# run tests
python -m pytest tests/ -v

# data migration (v0.2.x → v0.3.0 token IDs, file-based)
python scripts/migrate_to_v4.py --input-dir tokens_v3 --output-dir tokens_v4 --num-workers 16

# ── LMDB 标注管线 (v0.3.3, 从 LMDB 读 → 写回 LMDB) ──

# SSF annotation (三粒度 chroma 场: TonicField + LocalField + BeatField)
python scripts/generate_ssf.py annotate \
    --lmdb-path /root/autodl-tmp/data/processed/lmdb/chopinote_v4.lmdb \
    --num-workers 25

# Figuration annotation (11 种钢琴织体, per-voice per-bar)
python scripts/generate_fig.py annotate \
    --lmdb-path /root/autodl-tmp/data/processed/lmdb/chopinote_v4.lmdb \
    --num-workers 16

# Functional harmony annotation (v3: 尖锐模板 + ratio test, beat 级粒度)
python scripts/annotate_function.py annotate \
    --lmdb-path /root/autodl-tmp/data/processed/lmdb/chopinote_v4.lmdb \
    --num-workers 25

# SSF annotation audit (10-sample spot check)
python scripts/audit_ssf_sample.py \
    --lmdb-path /root/autodl-tmp/data/processed/lmdb/chopinote_v4.lmdb \
    --num-samples 10

# Fig + Func v3 combined audit (10 samples)
python scripts/audit_fig_func_sample.py \
    --lmdb-path /root/autodl-tmp/data/processed/lmdb/chopinote_v4.lmdb \
    --num-samples 10

# training (not yet run for v0.3.3)
python scripts/train/run_curriculum_training.py \
    --resume /root/autodl-tmp/chopinote/checkpoints/step_50000.pt \
    --batch-size 8 --phase1-steps 80000 --phase2-steps 40000
```

## Architecture

### v0.3.3 Core Design

**SSF (Sliding Scale Field)**: 12-dim tonic-anchored chroma field — three granularities (v0.3.3).
- **TonicField**: section-level PC histogram (归一化, max=1.0)
- **LocalField**: bar-level sparse delta from TonicField (threshold 0.15)
- **BeatField**: beat-level SSF per Position token (v0.3.3 新增)
- Position 0 = tonic, position i = i semitones from tonic (equivalence classes mod 12)
- Stored in LMDB as `v4:<file_id>:ssf` (key-mapped JSON, no sidecar files)
- Injected via `ssf_proj: nn.Linear(12, d_model)` into hidden state
- `SSFReconstructionHead`: d_model → 12 MSE regression (auxiliary task)

**Func v3 (Functional Harmony)**: Sharp chord-tone templates + ratio test.
- Templates: T(I)=pos[0,4,7], SD(IV)=pos[5,9,0], D(V)=pos[7,11,2], SDom(iv)=pos[5,8,0]
- Ratio test: `best_sim / second_best >= 1.20` (replaces sum-based confidence)
- Three granularities: section-level, bar-level (TonicField+LocalField+Markov), beat-level (primary)
- Beat-level uses local Markov chain (`_MARKOV_WEIGHT=0.2`) for transition smoothing
- Non-functional beats labeled `"non-func"` with reason (`low_sim` / `ambiguous` / `empty`)
- Stored in LMDB as `v4:<file_id>:func` (version 3 format)
- Avg 64.5% functional beats, 100% audit consistency

**Figuration (11 types)**: Per-voice 4-bar sliding window classification.
- Types: block, alberti, arpeggio, stride, octave_tremolo, walking_bass, countermelody, pedal, waltz, broken_octave, tremolo
- Stored in LMDB as `v4:<file_id>:fig`
- `fig_embedding: nn.Embedding(12, d_model)` — zero-init

**Cadence (5 types)**: `cadence_embedding: nn.Embedding(6, d_model)` — zero-init

**Four-Voice Time Slicing**: SATB via `<Voice 0>~<Voice 3>` tokens.
- 43 Program × 4 subtracks (172) + 4 Voice tokens (down from 512 Program tokens)
- `voice_embedding: nn.Embedding(5, d_model)` — zero-init per-voice identity
- `voice_same/samepos` bias (2 learnable scalars) — same-voice history + same-position cross-voice

**Vocabulary**: 929 → 542 → 574 tokens (v0.3.3)
- Removed: 30 Key, 30 Anticipate, 21 Chord func/7th/Inv, 508 unused Program
- Added: 12 Tonic, 4 Voice, 12 Fig, 5 Cadence

### LMDB Storage (v0.3.3)

统一 LMDB 存储，消除 ~480 万个 JSON 文件:
- **Path**: `/root/autodl-tmp/data/processed/lmdb/chopinote_v4.lmdb` (350GB map_size)
- **Key format**: `v4:<file_id>:<dtype>` where dtype ∈ {tokens, len, sec, ssf, fig, func, meta}
- **Files**: 1.6M, **Keys**: 11.2M across 7 data types
- **MVCC**: LMDB copy-on-write — concurrent readers OK, write transactions serialized
- **Fork safety**: Pool must be created BEFORE write_store; workers open read-only LMDB once in initializer
- **Chunked imap**: `WRITE_CHUNK=100000` prevents `pool.imap_unordered` internal queue explosion

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
  - `model.py` — MusicTransformer, 24 layers, 1.21B params, RoPE, **QK-Norm**, per-head scaling, **SSF field injection** (ssf_proj), **voice_embedding** (per-voice identity), **voice_bias** (same-voice/same-pos), **fig_embedding**, **cadence_embedding**, sec_bias (α/β/γ/δ), voice_count_embedding, measure_in_section_embedding, SectionPredictionHead, SSFReconstructionHead, weight tying, FP8 Linear, logit soft-capping.
  - `config.py` — ModelConfig (vocab_size=574, use_ssf, ssf_dim=12, n_voice_ids=5, n_fig_types=12, n_cadence_types=6), TrainingConfig (FP8, Z-loss, EMA, dropout schedule, SSF loss weight)
  - `train.py` — Trainer: next_token + sec_type(CE) + key_head(MSE) + SSF_recon(MSE) + Z-loss. No chord loss.
  - `dataset.py` — TokenDataset, supports LMDB mode (`use_lmdb=True`) with numpy→torch fast path. Auto-loads sec/ssf/fig/func from LMDB. Lengths from LMDB (8 bytes per lookup).
  - `generate.py` — **UNCHANGED from v0.2.x** (待 v0.3.4 框架-内容分离重写).

- **`chopinote_dataset/`** — Data:
  - `tokenizer.py` — REMITokenizer, **vocab=574** (grid_size=16, velocity_levels=8). TONIC_NAMES(12), VOICE_NAMES(4), PROGRAM_NAMES(43), FIGURATION_NAMES(12), CADENCE_NAMES(6).
  - `converter.py` — MusicXML/PDMX/MIDI→REMI.
  - `renderer.py` — REMI→MusicXML (fast path via ElementTree).
  - `lmdb_store.py` — LMDBStore: `get_tokens()` / `get_ssf()` / `get_func()` / `get_sec()` / `get_length()` / `get_raw()` / `_txn_put()`.

- **`chopinote_abc/`** — ABC Engine (原 `chopinote_evaluator/` 已删除, 逻辑合并):
  - `planner.py` — plan_structure, plan_harmony, **harmony_to_ssf()**, chord_func_to_ssf()
  - A1-A2-A3-B1-B2-C 六层全在此包

- **`scripts/`** — 标注 + 审计 + 数据:
  - `generate_ssf.py` — SSF 三粒度标注 (LMDB 读写, chunked imap, 进度文件)
  - `generate_fig.py` — Figuration 标注 (11 种织体, 4-bar 滑动窗口, per-voice)
  - `annotate_function.py` — Func v3 功能和声标注 (尖锐模板 + ratio test, beat 级)
  - `audit_ssf_sample.py` — SSF 审计: TonicField/LocalField/BeatField 独立重算验证
  - `audit_fig_func_sample.py` — Fig + Func v3 组合审计
  - `migrate_to_v4.py` — v0.2.x→v0.3.0 token ID 重映射
  - `migrate_to_lmdb.py` — JSON 文件→LMDB 导入
  - `classify_complexity.py` — F1-F5 复杂度自动分类
  - `annotate_sections_batch.py` / `annotate_sections_worker.py` — 段落标注
  - `structure_annotator.py` — 结构标注
  - `chord_annotator.py` / `parallel_chord_annotate.py` — 和弦标注 (旧)

### Key Constraints

- Vocab: 574 tokens, deterministic given grid_size=16, velocity_levels=8
- CloudTrain branch: local/remote diverged, write-only push only
- Git commits: `Derancarls <derancarls@foxmail.com>`, **NO** `Co-Authored-By`
- **禁止自动提交**: 未经用户明确要求，不得执行 `git commit`。只有用户明确说"提交"时才能提交
- **Commit message 版本号规范**: 同一版本内的多次修复用递增序号区分，如 `v0.3.3-fix1`, `v0.3.3-annotate1`。禁止无序号重复使用同一前缀
- `docs/` directory: NEVER delete from disk
- DataLoader: `num_workers=0` mandatory (multiprocessing crashes on long runs)
- bf16 training: no GradScaler needed
- LMDB: `map_size` 必须 ≥ 文件大小; pool 在 write_store 之前创建; worker initializer 每进程打开一次只读 LMDB
- generate.py: will be rewritten in v0.3.4 (framework-content separation)

### Design Documents (v0.3.x)

| Document | Topic |
|----------|-------|
| `docs/ssf_encoding_v0.3.x.md` | SSF tonic-anchored chroma field (三粒度, v0.3.3 新增 BeatField) |
| `docs/voice_time_slicing_v0.3.x.md` | Four-voice time slicing + Voice bias |
| `docs/figuration_encoding_v0.3.x.md` | Piano figuration types (11 classes) |
| `docs/framework_content_separation_v0.3.x.md` | Framework/content separation (v0.3.1) |
| `docs/cadence_awareness_v0.3.x.md` | Cadence zone + embedding |
| `docs/phrase_layer_design_v0.3.x.md` | Phrase layer design + v0.3.x adaptations |
| `docs/duration_saturation_v0.3.x.md` | Duration saturation — 17 buckets, B1 guard |
| `docs/voice_splitting_v0.3.x.md` | Piano 2-track → 4-voice split + VoicePlan |
| `docs/curriculum_training_v0.3.x.md` | F1-F5 quality filter + 4-metric classification + 3-phase curriculum |
| `docs/ssf_pairwise_bias_backup.md` | SSF pairwise similarity bias (backup) |
| `docs/remi_z_track_continuous.md` | REMI-z track-continuous interleaving (backup) |
| `docs/v0.2.6_audit_v0.3.x_gap.md` | v0.2.6 pain points × v0.3.x coverage |
| `docs/ROADMAP.md` | 完整路线图、依赖拓扑、当前状态 |
