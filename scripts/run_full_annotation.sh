#!/bin/bash
set -e
LOG=/root/autodl-tmp/pipeline_rerun.log
DIR=/root/autodl-tmp/data/processed/tokens_v4
echo "[$(date)] ===== 全量标注管线 =====" | tee -a $LOG
echo "[$(date)] Stage 1/3: SSF → .ssf.json" | tee -a $LOG
python /root/Chopinote-AI/scripts/generate_ssf.py annotate --input-dir $DIR --num-workers 25 >> $LOG 2>&1
echo "[$(date)] SSF done: $(find $DIR -name '*.ssf.json' | wc -l) files" | tee -a $LOG

echo "[$(date)] Stage 2/3: Fig → inject <FigV> into .tokens" | tee -a $LOG
python /root/Chopinote-AI/scripts/generate_fig.py annotate --input-dir $DIR --num-workers 8 >> $LOG 2>&1
echo "[$(date)] Fig done" | tee -a $LOG

echo "[$(date)] Stage 3/3: Func → .func.json" | tee -a $LOG
python /root/Chopinote-AI/scripts/annotate_function.py annotate --input-dir $DIR --num-workers 25 >> $LOG 2>&1
echo "[$(date)] Func done: $(find $DIR -name '*.func.json' | wc -l) files" | tee -a $LOG

echo "[$(date)] ===== 管线完成 =====" | tee -a $LOG
