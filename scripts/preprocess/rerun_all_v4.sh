#!/bin/bash
# v0.3.2 预处理完整管线: PDMX → MusicXML → MIDI → 标注
# 后台: nohup bash scripts/preprocess/rerun_all_v4.sh &> /root/autodl-tmp/preprocess_v4.log &
set -e
cd /root/Chopinote-AI
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "============================================================"
log "预处理 v4 启动 (v0.3.2 gen5)"
log "============================================================"

# ── Stage 1: PDMX ───────────────────────────────────────────
log "Stage 1/6: PDMX → tokens_v4"
python scripts/preprocess/rerun_pdmx.py 2>&1 || log "PDMX exit=$? (non-fatal)"
log "Stage 1/6 done ($(date '+%H:%M:%S'))"

# ── Stage 2: MusicXML ───────────────────────────────────────
log "Stage 2/6: MusicXML → tokens_v4"
python scripts/preprocess/rerun_musicxml.py 2>&1 || log "MusicXML exit=$? (non-fatal)"
log "Stage 2/6 done ($(date '+%H:%M:%S'))"

# ── Stage 3: MIDI ───────────────────────────────────────────
log "Stage 3/6: MIDI → tokens_v4 (25 workers)"
python scripts/preprocess/run_fast_preprocess.py 2>&1 || log "MIDI exit=$? (non-fatal)"
log "Stage 3/6 done ($(date '+%H:%M:%S'))"

# Count tokens so far
TOKEN_COUNT=$(find /root/autodl-tmp/data/processed/tokens_v4 -name "*.tokens" | wc -l)
log "Total .tokens files after conversion: $TOKEN_COUNT"

# ── Stage 4: Structure annotation ───────────────────────────
log "Stage 4/6: Structure annotation → .sec.json"
python scripts/structure_annotator.py annotate \
    --input-dir /root/autodl-tmp/data/processed/tokens_v4 \
    --output-dir /root/autodl-tmp/data/processed/tokens_v4 \
    --num-workers 25 2>&1 || log "Structure exit=$? (non-fatal)"
log "Stage 4/6 done ($(date '+%H:%M:%S'))"

# ── Stage 5: SSF annotation ─────────────────────────────────
log "Stage 5/6: SSF annotation → .ssf.json"
python scripts/generate_ssf.py annotate \
    --input-dir /root/autodl-tmp/data/processed/tokens_v4 \
    --num-workers 25 2>&1 || log "SSF exit=$? (non-fatal)"
log "Stage 5/6 done ($(date '+%H:%M:%S'))"

# ── Stage 6: Figuration annotation ──────────────────────────
log "Stage 6/6: Figuration annotation → inject FigV tokens"
python scripts/generate_fig.py annotate \
    --input-dir /root/autodl-tmp/data/processed/tokens_v4 \
    --num-workers 25 2>&1 || log "Fig exit=$? (non-fatal)"
log "Stage 6/6 done ($(date '+%H:%M:%S'))"

# ── Dedup + Split ───────────────────────────────────────────
log "Dedup + split: train/val/test.txt"
python scripts/preprocess/dedup_and_split.py \
    --input-dir /root/autodl-tmp/data/processed/tokens_v4 2>&1 || log "Dedup exit=$? (non-fatal)"
log "Dedup done ($(date '+%H:%M:%S'))"

# ── Summary ─────────────────────────────────────────────────
TOTAL_TOKENS=$(find /root/autodl-tmp/data/processed/tokens_v4 -name "*.tokens" | wc -l)
TOTAL_SEC=$(find /root/autodl-tmp/data/processed/tokens_v4 -name "*.sec.json" | wc -l)
TOTAL_SSF=$(find /root/autodl-tmp/data/processed/tokens_v4 -name "*.ssf.json" | wc -l)
SIZE=$(du -sh /root/autodl-tmp/data/processed/tokens_v4/ 2>/dev/null | cut -f1)
log "============================================================"
log "预处理 v4 完成 $(date)"
log "  .tokens:   $TOTAL_TOKENS"
log "  .sec.json: $TOTAL_SEC"
log "  .ssf.json: $TOTAL_SSF"
log "  Size:      $SIZE"
log "============================================================"
