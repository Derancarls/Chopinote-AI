#!/bin/bash
# v0.3.3 标注管线: SSF → Figuration → Function → Dedup/Split → Token lengths
# nohup bash scripts/preprocess/run_annotations.sh &> /root/autodl-tmp/annotations.log &
set -e
cd /root/Chopinote-AI
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "============================================================"
log "标注管线启动 (SSF → Fig → Func → Split → Lengths)"
log "============================================================"

# ── Stage 1: SSF annotation (含 beat_fields) ─────────────────
log "Stage 1/5: SSF annotation → .ssf.json (含 beat_fields)"
python scripts/generate_ssf.py annotate \
    --input-dir /root/autodl-tmp/data/processed/tokens_v4 \
    --num-workers 25 2>&1 || log "SSF exit=$? (non-fatal)"
log "Stage 1/5 done ($(date '+%H:%M:%S'))"

SSF_COUNT=$(find /root/autodl-tmp/data/processed/tokens_v4 -name "*.ssf.json" | wc -l)
log ".ssf.json files: $SSF_COUNT"

# ── Stage 2: Figuration annotation ───────────────────────────
log "Stage 2/5: Figuration annotation → inject FigV tokens"
python scripts/generate_fig.py annotate \
    --input-dir /root/autodl-tmp/data/processed/tokens_v4 \
    --num-workers 25 2>&1 || log "Fig exit=$? (non-fatal)"
log "Stage 2/5 done ($(date '+%H:%M:%S'))"

# ── Stage 3: Function annotation (读 SSF, 三粒度) ────────────
log "Stage 3/5: Function annotation → .func.json (三粒度)"
python scripts/annotate_function.py annotate \
    --input-dir /root/autodl-tmp/data/processed/tokens_v4 \
    --num-workers 25 2>&1 || log "Func exit=$? (non-fatal)"
log "Stage 3/5 done ($(date '+%H:%M:%S'))"

FUNC_COUNT=$(find /root/autodl-tmp/data/processed/tokens_v4 -name "*.func.json" | wc -l)
log ".func.json files: $FUNC_COUNT"

# ── Stage 4: Dedup + Split ───────────────────────────────────
log "Stage 4/5: Dedup + split → train/val/test.txt"
python scripts/preprocess/dedup_and_split.py \
    --input-dir /root/autodl-tmp/data/processed/tokens_v4 2>&1 || log "Dedup exit=$? (non-fatal)"
log "Stage 4/5 done ($(date '+%H:%M:%S'))"

# ── Stage 5: Token lengths ───────────────────────────────────
log "Stage 5/5: Generating token_lengths.json"
python scripts/preprocess/generate_token_lengths.py 2>&1 || log "Lengths exit=$? (non-fatal)"
log "Stage 5/5 done ($(date '+%H:%M:%S'))"

# ── Summary ───────────────────────────────────────────────────
TOTAL_TOKENS=$(find /root/autodl-tmp/data/processed/tokens_v4 -name "*.tokens" | wc -l)
TOTAL_SSF=$(find /root/autodl-tmp/data/processed/tokens_v4 -name "*.ssf.json" | wc -l)
TOTAL_FUNC=$(find /root/autodl-tmp/data/processed/tokens_v4 -name "*.func.json" | wc -l)
SIZE=$(du -sh /root/autodl-tmp/data/processed/tokens_v4/ 2>/dev/null | cut -f1)

log "============================================================"
log "标注管线完成 $(date)"
log "  .tokens:   $TOTAL_TOKENS"
log "  .ssf.json: $TOTAL_SSF"
log "  .func.json: $TOTAL_FUNC"
log "  Size:      $SIZE"
log "============================================================"
