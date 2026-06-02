# Chopinote-AI

> Give it a few bars — it finishes the piece.

Chopinote-AI is a **decoder-only Transformer** (1.21B parameters) that composes classical piano music in the style of your input. Feed it the first few measures of a piece (MusicXML), and it generates a stylistically coherent continuation — with structural awareness of musical form, tonal harmony via SSF encoding, and a six-layer ABC cognitive engine for real-time quality control.

Output is standard MusicXML, editable in MuseScore, Finale, Sibelius, or any notation software.

---

## Why Chopinote-AI?

### 1. SSF (Sliding Scale Field) — Continuous Tonal Encoding

v0.3.0 replaces discrete Key/Chord tokens with a **12-dimensional tonic-anchored chroma field** that continuously represents harmonic context:

| Field | Scope | Content |
|-------|-------|---------|
| **TonicField** | Section-level | 12-dim chroma vector, tonic at position 0 |
| **LocalField** | Bar-level | Sparse delta from TonicField (chroma shift) |

- No discrete key/chord vocabulary — the field is injected directly into the hidden state via `ssf_proj: nn.Linear(12, d_model)`
- `SSFReconstructionHead` provides auxiliary MSE regression during training
- Modulation, chromaticism, and harmonic ambiguity are encoded naturally as continuous vectors

### 2. Four-Voice SATB Time Slicing

Four simultaneous voices (Soprano/Alto/Tenor/Bass) modeled with per-voice identity:

| Mechanism | Role |
|-----------|------|
| **Voice Embedding** | Per-voice identity (zero-init for training stability) |
| **Voice Bias** | same-voice history attraction + same-position cross-voice coordination |
| **Voice tokens** | `<Voice 0>`–`<Voice 3>` demarcate voice transitions in the token stream |

43 Program × 4 subtracks = 172 entities, with explicit voice awareness replacing 512 discrete Program tokens.

### 3. Explicit Figuration Encoding — 11 Piano Textures

Piano-specific figuration types are explicitly encoded:

```
Alberti bass, arpeggio, broken chord, chordal, octave, scale,
tremolo, trill, repeated notes, melody+accompaniment, polyphonic
```

`fig_embedding: nn.Embedding(12, d_model)` — zero-init, injected at Figuration token positions.

### 4. Cadence Awareness — 5 Types

Five cadence types with dedicated embedding + zone-based SSF boost:

| Type | Description |
|------|-------------|
| **PAC** | Perfect Authentic Cadence |
| **IAC** | Imperfect Authentic Cadence |
| **HC** | Half Cadence |
| **DC** | Deceptive Cadence |
| **PC** | Plagal Cadence |

### 5. DurSat (Duration Saturation) — v0.3.1

Per-voice cumulative duration tracking encoded as 17 saturation buckets (0/16 ~ 16/16), injected at Position tokens only. B1 hard constraints prevent duration overflow.

### 6. ABC Engine — Six-Layer Cognitive Architecture

```
A1 (框架记忆) → A2 (动机提取) → A3 (统计画像) → B1 (硬约束) → B2 (决策调参) → C (评价层)
```

| Layer | Role |
|-------|------|
| **A1 — Perception: Framework** | Section planning, harmonic progression, structure memory |
| **A2 — Perception: Motif** | Seed analysis, motif DNA extraction, landmark management |
| **A3 — Perception: Statistics** | Per-bar statistics, section snapshots, baseline comparison |
| **B1 — Decision: Hard Bans** | Voice range, parallel intervals, duration overflow, note density |
| **B2 — Decision: Tuning** | Temperature zones, innovation budget, parameter adjustment |
| **C — Evolution** | MusicXML review, token↔XML comparison, DPO preference learning |

The ABC Engine replaces the old four-stage feedback loop with a unified perception→decision→evolution pipeline. All evaluation (legality, statistics, theory, coherence) is integrated into the C layer with 20+ metrics.

### 7. Section-Aware Attention — Musical Form as Bias

Learned section biases model formal structure:

| Bias | Role |
|------|------|
| **α** | Same-section boost — notes within the same section attend more strongly |
| **β** | Bar-distance decay — nearby bars matter more than distant ones |
| **γ** | Section-type affinity — similar section types attend to each other |
| **δ** | Section-boundary reset — attention resets at section boundaries |

### 8. Multi-Task Learning — Auxiliary Structural Supervision

| Task | Head | What It Learns |
|------|------|----------------|
| **Next-token prediction** | LM head | Note-by-note fluency and style |
| **Section prediction** | SectionPredictionHead | Bar boundaries, key changes, section types |
| **SSF reconstruction** | SSFReconstructionHead | 12-dim chroma field (auxiliary MSE) |

### 9. Memory-Efficient Architecture

- **QK-Norm** + per-head scaling for training stability
- **Bias recomputation inside gradient checkpointing** — biases recomputed from raw data (~1 MiB) instead of stored as full attention masks
- **Bias detach** — combined bias detached before SDPA
- **RoPE** — 4.1× faster than ALiBi with equivalent quality
- **FP8 Linear** — Blackwell native FP8 inference
- **BF16 autocast** — no GradScaler needed

---

## Architecture

```
Input Token IDs (B, T)
    │
    ▼
┌─────────────────────┐
│  Token Embedding     │  d_model=2048
│  + RoPE              │
│  + Bar Embedding     │
│  + Voice Embedding   │  per-voice identity (SATB)
└─────────┬───────────┘
          │
┌─────────▼──────────────────────────────────┐
│           24× TransformerBlock             │
│  ┌─────────────────────────────────────┐   │
│  │  Multi-Head Self-Attention           │   │
│  │  • RoPE + QK-Norm                   │   │
│  │  • Section bias (α/β/γ/δ)           │   │
│  │  • Voice bias (same-voice/same-pos) │   │
│  │  • Flash / mem-efficient attention  │   │
│  └──────────────┬──────────────────────┘   │
│  ┌──────────────▼──────────────────────┐   │
│  │  SSF Field Injection                │   │
│  │  • ssf_proj(tonic_field + local_delta)│  │
│  │  • Voice Embedding                  │   │
│  │  • Figuration Embedding             │   │
│  │  • Cadence Embedding                │   │
│  │  • DurSat Embedding (v0.3.1)        │   │
│  └──────────────┬──────────────────────┘   │
│  ┌──────────────▼──────────────────────┐   │
│  │  Position-wise FFN (d_ff=8192)       │   │
│  │  Residual + LayerNorm                │   │
│  └─────────────────────────────────────┘   │
└─────────┬──────────────────────────────────┘
          │
          ▼
┌─────────────────────┐  ┌──────────────────────┐
│  LM Head             │  │  SectionPredictionHead │
│  (next token logits)  │  │  (bars/key/type)      │
└─────────────────────┘  └──────────────────────┘
┌─────────────────────┐
│  SSFReconstructionHead │  12-dim MSE regression
└─────────────────────┘
```

### Specs

| Parameter | Value |
|-----------|-------|
| Parameters | 1.21B |
| Layers | 24 |
| Attention heads | 32 |
| d_model | 2048 |
| d_ff | 8192 |
| Vocab size | **542** (v0.3.0: 929 → 542) |
| Context length | 4096 tokens |
| Position encoding | RoPE (θ=10000) |
| Precision | BF16 training, FP8/BF16 inference |
| Peak VRAM (training, bs=8) | ~22 GiB |
| Checkpoint size | ~4.5 GB |

### Tokenization (REMI v4)

v0.3.0 tokenizer: 542 tokens (grid_size=16, velocity_levels=8)

| Category | Tokens | Notes |
|----------|--------|-------|
| Special | PAD, BOS, EOS, MASK | 4 |
| Structure | Bar, Section, SecSum | 3 |
| Tonic | 12 tonics (`<Tonic C>` ~ `<Tonic B>`) | 12 |
| Voice | 4 voices (SATB) | 4 |
| Position | 16 grid positions | 16 |
| Note_ON | 12 pitch classes × 8 octaves = 96 | 96 |
| Velocity | 8 levels | 8 |
| Duration | 1–16 sixteenths | 16 |
| Program | 43 programs × 4 subtracks = 172 | 172 |
| TimeSig | 14 signatures | 14 |
| Tempo | 30–260 BPM in 10-bpm steps | 24 |
| Figuration | 11 types + none | 12 |
| Cadence | 5 types + none | 6 |
| Markings | Clef, dynamic, hairpin, articulation, ornament, pedal, etc. | ~70 |
| Other | Rest, Beat, Tuplet, Octave, Arpeggio, Bass, Repeat, Jump | ~85 |

**Removed from v0.2.x**: 30 Key tokens, 30 Anticipate, 21 Chord (func/7th/Inv), 508 unused Program — replaced by SSF, Voice, Tonic, Figuration, Cadence.

---

## ABC Engine

### A — Perception Layer

| Subsystem | Function |
|-----------|----------|
| **A1 — Framework Memory** | Section planning, harmonic progression (→ SSF via `harmony_to_ssf()`), framework fallback |
| **A2 — Motif Extraction** | Seed analysis, motif DNA (contour, rhythm, register) extraction, landmark detection |
| **A3 — Statistical Profiling** | Baseline stats, per-bar density/rest_ratio, section snapshots, trend detection, duration tracking (DurSat) |

### B — Decision Layer

| Subsystem | Function |
|-----------|----------|
| **B1 — Hard Constraints** | Context bans, voice range, parallel intervals, duration overflow guard, note density caps |
| **B2 — Parameter Tuning** | Temperature zone annealing, innovation budget, fatal signal detection |

### C — Evolution Layer

| Component | Function |
|-----------|----------|
| **MusicXML Review** | Legality check, theory evaluation (20+ rules), score-level analysis |
| **Token↔XML Comparison** | Cross-modal verification |
| **DPO Preference Learning** | Automatic preference pair collection → LoRA fine-tuning |

---

## Data Pipeline

### Preprocessing

```
Raw files (MIDI + PDMX + MusicXML) → REMI tokenization → tokens_v4
    → SSF annotation (.ssf.json)         # generate_ssf.py
    → Figuration annotation (.fig.json)   # generate_fig.py
    → Structure annotation (.sec.json)    # structure_annotator.py
    → Quality filtering (F1–F5)          # classify_complexity.py
    → Complexity classification (L1–L5)
    → Train/Val split
```

### Data Scale

| Type | Source | Files |
|------|--------|-------|
| MIDI | MAESTRO, Lakh, GiantMIDI, POP909, MusicNet, EMOPIA | ~1.37M |
| PDMX | Pop music scores | ~250K |
| MusicXML | ASAP, ATEPP, Openscore, internal | ~4.1K |
| **Total** | | **~1.62M** |

---

## Training

### Three-Phase Curriculum (v0.3.1)

| Phase | Data | Steps | Focus |
|-------|------|-------|-------|
| 1 — Grammar | L1–L2 (simple) | 40K | Note fluency, basic voice leading |
| 2 — Structure | L1–L4 (mixed) | 50K | Formal structure, harmonic progression |
| 3 — Refinement | L1–L5 (all) | 30K | Articulation, expression, cadence quality |

Gate mechanism: each phase auto-advances when validation loss plateaus or step target reached.

### Hardware

- **GPU**: RTX 5090 — 32 GB VRAM
- **Batch**: bs=8, grad_accum=4 (effective bs=32)
- **Data**: ~1.62M files, ~13.7B tokens, ~400 GB on disk
- **Speed**: ~5–6 s/step on RTX 5090 (bf16)

---

## Project Structure

```
chopinote_model/        Core model and training
├── model.py            MusicTransformer (24 layers, RoPE, SSF, Voice, Fig, Cadence, DurSat)
├── config.py           ModelConfig, TrainingConfig, PhaseConfig
├── train.py            Trainer, multi-task loss (next-token + section + SSF_recon + Z-loss)
├── dataset.py          TokenDataset with sidecar auto-loading (.ssf.json, .sec.json)
├── generate.py         Generation (⚠ deprecated, v0.3.1 rewrite pending)
├── fp8_linear.py       FP8 mixed-precision linear layer
├── auto_config.py      Hardware detection + optimal inference/training config

chopinote_abc/          ABC Engine (A1/A2/A3 → B1/B2 → C)
├── database.py         A1DB (framework), A2DB (motif), A3DB (statistics)
├── planner.py          Structure + harmony planning, SSF conversion, cadence boost
├── motif.py            Motif DNA extraction, contour ops (invert, fragment, diminish)
├── constraints.py      B1 theory rules (voice range, parallel intervals)
├── metrics.py          20+ token/score-level metrics
├── scoring.py          C layer evaluation (intrinsic + consistency)
├── decision.py         B1 hard bans + B2 parameter tuning
├── parser.py           MusicXML parser (Score, Measure, Note)
├── logging.py          ABC generation logger + summary

chopinote_dataset/      Data processing
├── tokenizer.py        REMITokenizer (vocab=542, grid_size=16)
├── converter.py        MusicXML/PDMX/MIDI → REMI tokens
├── fast_converter.py   mido-based ~80× faster MIDI conversion
├── renderer.py         REMI → MusicXML (ElementTree fast-path, 54000× speedup)
├── processor.py        MusicXML preprocessor

chopinote_cli/          CLI entry point
├── main.py             Generation + evaluation CLI
├── config.py           Config loading, validation, auto-discovery
├── presets.py          Style presets (romantic, baroque, classical, etc.)
└── generation_config.yaml  Built-in defaults

scripts/                Utility scripts
├── migrate_to_v4.py    v0.2.x → v0.3.0 token ID remapping
├── generate_ssf.py     SSF annotation (multiprocessing)
├── generate_fig.py     Figuration annotation (multiprocessing)
├── structure_annotator.py  Structure annotation
├── classify_complexity.py  F1–F5 filtering + 4-metric classification + split
├── train/              Training launcher, DPO, curriculum control
├── generate/           Batch generation, roundtrip testing
└── analysis/           Token analysis, verification

docs/                   Design documents (v0.3.x)
├── ssf_encoding_v0.3.x.md
├── voice_time_slicing_v0.3.x.md
├── figuration_encoding_v0.3.x.md
├── cadence_awareness_v0.3.x.md
├── duration_saturation_v0.3.x.md
├── voice_splitting_v0.3.x.md
├── framework_content_separation_v0.3.x.md
├── phrase_layer_design_v0.3.x.md
├── curriculum_training_v0.3.x.md
└── ...
```

---

## Quick Start

```bash
# Install
pip install -e .

# Continue a piece
chopin checkpoints/step_N.pt input.musicxml -o output.musicxml

# Generate multiple variants
chopin checkpoints/step_N.pt input.musicxml -n 5

# Custom config
chopin checkpoints/step_N.pt input.musicxml --config my_cfg.yaml --max-bars 64
```

---

## Key Design Documents

| Document | Topic |
|----------|-------|
| `docs/ssf_encoding_v0.3.x.md` | SSF tonic-anchored chroma field |
| `docs/voice_time_slicing_v0.3.x.md` | Four-voice SATB + Voice bias |
| `docs/figuration_encoding_v0.3.x.md` | 11 piano figuration types |
| `docs/cadence_awareness_v0.3.x.md` | Cadence zone + embedding |
| `docs/duration_saturation_v0.3.x.md` | 17-bucket DurSat + B1 guard |
| `docs/voice_splitting_v0.3.x.md` | Piano 2-track → 4-voice split |
| `docs/framework_content_separation_v0.3.x.md` | Framework/content separation (v0.3.2) |
| `docs/curriculum_training_v0.3.x.md` | F1–F5 filter + 5-level classification + 3-phase curriculum |
| `docs/abc_engine.md` | ABC Engine (A/B/C architecture) |

---

## Version History

| Version | Key Changes |
|---------|-------------|
| **v0.2.6** | Section-aware attention, chord bias, 929 vocab |
| **v0.3.0** | SSF encoding, Voice SATB, Figuration, Cadence, 542 vocab, RoPE, ABC Engine |
| **v0.3.1** | Voice splitting, DurSat, F1–F5 filtering, classify_complexity, cadence boost |

---

*Chopinote-AI — 让古典音乐创作从灵感开始，而不是从空白五线谱开始。*
