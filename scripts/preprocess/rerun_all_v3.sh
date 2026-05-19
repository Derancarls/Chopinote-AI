#!/bin/bash
# 三格式预处理流水线 (v3): PDMX → MusicXML → MIDI
# 后台: nohup bash scripts/rerun_all_v3.sh &> /root/autodl-tmp/preprocess_v3.log &
cd /root/Chopinote-AI
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "============================================================"
log "预处理 v3 启动"
log "============================================================"

log "Stage 1/3: PDMX"
python scripts/rerun_pdmx.py 2>&1 || log "PDMX exit=$?"
log "Stage 1/3 done"

log "Stage 2/3: MusicXML"
python scripts/rerun_musicxml.py 2>&1 || log "MusicXML exit=$?"
log "Stage 2/3 done"

log "Stage 3/3: MIDI"
python scripts/run_fast_preprocess.py 2>&1 || log "MIDI exit=$?"
log "Stage 3/3 done"

log "============================================================"
log "预处理 v3 完成 $(date)"
log "============================================================"
