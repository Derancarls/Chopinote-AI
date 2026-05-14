#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────
# Chopinote-AI 训练启动器 + 看门狗
# 用法:
#   ./scripts/run_training.sh                      # 默认 Phase 1 (MIDI)
#   ./scripts/run_training.sh --phase2             # Phase 1 + Phase 2
#   ./scripts/run_training.sh --watchdog-only      # 仅启动看门狗（恢复已中断的训练）
#   ./scripts/run_training.sh --status             # 查看训练状态
#   ./scripts/run_training.sh --stop               # 停止训练
# ──────────────────────────────────────────────

SESSION_NAME="chopinote-train"
LOCK_FILE="/tmp/chopinote-train.lock"
PID_FILE="/tmp/chopinote-train.pid"
LOG_DIR="${CHOPINOTE_LOG_DIR:-/root/autodl-tmp/chopinote/logs}"
DATA_DIR="${CHOPINOTE_DATA_DIR:-/root/autodl-tmp/data/processed}"
CHECKPOINT_DIR="${CHOPINOTE_OUTPUT_DIR:-/root/autodl-tmp/chopinote/checkpoints}"
TENSORBOARD_DIR="${CHOPINOTE_TB_DIR:-/root/autodl-tmp/chopinote/tensorboard}"
WATCHDOG_LOG="${LOG_DIR}/watchdog.log"
CRASH_LOG="${LOG_DIR}/crashes.log"
MAX_RESTARTS=5
RESTART_DELAY=20        # 崩溃后等待秒数

mkdir -p "$LOG_DIR" "$TENSORBOARD_DIR"

# ── 状态/停止子命令 ───────────────────────────
if [[ "${1:-}" == "--status" ]]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "✅ 训练会话运行中 (tmux: $SESSION_NAME)"
        echo "   附加: tmux attach -t $SESSION_NAME"
        echo "   日志: $LOG_DIR"
        echo "   TB:   tensorboard --logdir $TENSORBOARD_DIR"
        tail -5 "$WATCHDOG_LOG" 2>/dev/null || true
    else
        echo "❌ 训练未运行"
    fi
    exit 0
fi

if [[ "${1:-}" == "--stop" ]]; then
    echo "正在停止训练..."
    tmux send-keys -t "$SESSION_NAME" C-c 2>/dev/null || true
    sleep 5
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    rm -f "$LOCK_FILE" "$PID_FILE"
    echo "已停止"
    exit 0
fi

# ── 单实例锁 ──────────────────────────────────
if [[ -f "$LOCK_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "❌ 训练已在运行 (PID $OLD_PID)"
        echo "   查看状态: $0 --status"
        exit 1
    fi
    echo "⚠️  发现残留锁文件，清除"
    rm -f "$LOCK_FILE" "$PID_FILE"
fi
echo $$ > "$PID_FILE"
trap 'rm -f "$LOCK_FILE" "$PID_FILE"' EXIT

# ── 性能调优环境变量 ──────────────────────────
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_CUDNN_V8_API_ENABLED=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128   # 减少碎片化
export TOKENIZERS_PARALLELISM=false                    # 单 worker 不需要 tokenizer 并行
export TORCH_LOGS=""

cd /root/Chopinote-AI

# ── 训练参数 ──────────────────────────────────
PHASE1_STEPS=120000
PHASE1_LR=1.5e-4
PHASE1_WARMUP=4000
PHASE2_STEPS=50000
PHASE2_LR=1e-4
PHASE2_WARMUP=2000
BATCH_SIZE=2

# 检查最新 checkpoint 用于恢复
LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/step_*.pt 2>/dev/null | head -1)
RESUME_ARG=""
if [[ -n "$LATEST_CKPT" ]]; then
    LATEST_STEP=$(basename "$LATEST_CKPT" | sed 's/step_//;s/\.pt//')
    if [[ "$LATEST_STEP" -gt 0 ]] 2>/dev/null; then
        RESUME_ARG="--resume $LATEST_CKPT"
        echo "📌 发现已有 checkpoint (step $LATEST_STEP)，将断点续训"
    fi
fi

# ── 看门狗（tmux 内部执行） ──────────────────
WATCHDOG_CODE='
set -euo pipefail
cd /root/Chopinote-AI

RESTART_COUNT=0
PHASE2_FLAG='${2:+--musicxml-train-list "$DATA_DIR/train.txt"}'

while true; do
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] 🚀 训练启动 (重启 #$RESTART_COUNT)" >> '"$LOG_DIR"'/watchdog.log

    # 将 stderr 同时写入崩溃日志
    python scripts/run_curriculum_training.py \
        --midi-train-list "$DATA_DIR/train.txt" \
        --musicxml-train-list "$DATA_DIR/train.txt" \
        --val-list "$DATA_DIR/val.txt" \
        --phase1-steps '"$PHASE1_STEPS"' \
        --phase1-lr '"$PHASE1_LR"' \
        --phase1-warmup '"$PHASE1_WARMUP"' \
        --phase2-steps '"$PHASE2_STEPS"' \
        --phase2-lr '"$PHASE2_LR"' \
        --phase2-warmup '"$PHASE2_WARMUP"' \
        --batch-size '"$BATCH_SIZE"' \
        --output-dir '"$CHECKPOINT_DIR"' \
        '"$RESUME_ARG"' \
        2>> '"$CRASH_LOG"'

    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] ✅ 训练正常完成" >> '"$LOG_DIR"'/watchdog.log
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ❌ 训练崩溃 (exit=$EXIT_CODE, 重启 #$RESTART_COUNT)" >> '"$LOG_DIR"'/watchdog.log

    if [[ $RESTART_COUNT -ge '"$MAX_RESTARTS"' ]]; then
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] ⛔ 已达最大重启次数 ($MAX_RESTARTS)，停止" >> '"$LOG_DIR"'/watchdog.log
        break
    fi

    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ⏳ 等待 ${RESTART_DELAY}s 后重启..." >> '"$LOG_DIR"'/watchdog.log
    sleep '"$RESTART_DELAY"'
done
'

# ── 启动 tmux ────────────────────────────────
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
fi

# 检测 --phase2 标记
TMUX_ENV=""
if [[ "${1:-}" == "--phase2" ]]; then
    TMUX_ENV="PHASE2=1"
fi

tmux new-session -d -s "$SESSION_NAME" -x 160 -y 50
tmux send-keys -t "$SESSION_NAME" "cd /root/Chopinote-AI" Enter

# 注入环境变量 + 看门狗
tmux send-keys -t "$SESSION_NAME" \
    "export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; " \
    "export TORCH_CUDNN_V8_API_ENABLED=1; " \
    "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128; " \
    "export TOKENIZERS_PARALLELISM=false; " \
    "export TF_ENABLE_ONEDNN_OPTS=0" Enter

# 写入看门狗脚本到临时文件（避免 tmux 引用问题）
cat > /tmp/chopinote-watchdog.sh << 'WDEOL'
#!/bin/bash
set -euo pipefail

RESTART_COUNT=0
MAX_RESTARTS=5
RESTART_DELAY=20
LOG_DIR="${CHOPINOTE_LOG_DIR:-/root/autodl-tmp/chopinote/logs}"
TENSORBOARD_DIR="${CHOPINOTE_TB_DIR:-/root/autodl-tmp/chopinote/tensorboard}"
DATA_DIR="${CHOPINOTE_DATA_DIR:-/root/autodl-tmp/data/processed}"
CHECKPOINT_DIR="${CHOPINOTE_OUTPUT_DIR:-/root/autodl-tmp/chopinote/checkpoints}"
PHASE1_STEPS=120000
PHASE1_LR=1.5e-4
PHASE1_WARMUP=4000
PHASE2_STEPS=50000
PHASE2_LR=1e-4
PHASE2_WARMUP=2000
BATCH_SIZE=2

# 检测最新 checkpoint 用于恢复
LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/step_*.pt 2>/dev/null | head -1)
RESUME_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    LATEST_STEP=$(basename "$LATEST_CKPT" | sed 's/step_//;s/\.pt//')
    if [ "$LATEST_STEP" -gt 0 ] 2>/dev/null; then
        RESUME_ARG="--resume $LATEST_CKPT"
    fi
fi

cd /root/Chopinote-AI

while true; do
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] 🚀 训练启动 (重启 #$RESTART_COUNT)" >> "$LOG_DIR"/watchdog.log

    # 设置崩溃日志重定向（tee 同时输出到终端和文件）
    python scripts/run_curriculum_training.py \
        --midi-train-list "$DATA_DIR/train.txt" \
        --musicxml-train-list "$DATA_DIR/train.txt" \
        --val-list "$DATA_DIR/val.txt" \
        --phase1-steps $PHASE1_STEPS \
        --phase1-lr $PHASE1_LR \
        --phase1-warmup $PHASE1_WARMUP \
        --phase2-steps $PHASE2_STEPS \
        --phase2-lr $PHASE2_LR \
        --phase2-warmup $PHASE2_WARMUP \
        --batch-size $BATCH_SIZE \
        --output-dir "$CHECKPOINT_DIR" \
        --log-dir "$TENSORBOARD_DIR" \
        $RESUME_ARG \
        2>> "$LOG_DIR"/crashes.log

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] ✅ 训练正常完成" >> "$LOG_DIR"/watchdog.log
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ❌ 训练崩溃 (exit=$EXIT_CODE, 重启 #$RESTART_COUNT)" >> "$LOG_DIR"/watchdog.log

    if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] ⛔ 已达最大重启次数 ($MAX_RESTARTS)，停止" >> "$LOG_DIR"/watchdog.log
        break
    fi

    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ⏳ 等待 ${RESTART_DELAY}s 后重启..." >> "$LOG_DIR"/watchdog.log
    sleep $RESTART_DELAY
done
WDEOL

chmod +x /tmp/chopinote-watchdog.sh
tmux send-keys -t "$SESSION_NAME" "bash /tmp/chopinote-watchdog.sh" Enter

# 写锁
echo $$ > "$LOCK_FILE"
echo "✅ 训练已启动 (tmux 会话: $SESSION_NAME)"
echo ""
echo "   附加:         tmux attach -t $SESSION_NAME"
echo "   分离:         Ctrl+B, D"
echo "   查看日志:     tail -f $LOG_DIR/watchdog.log"
echo "   TensorBoard:  tensorboard --logdir $TENSORBOARD_DIR"
echo "   停止训练:     $0 --stop"
echo "   查看状态:     $0 --status"
echo "   崩溃记录:     $LOG_DIR/crashes.log"
echo ""
echo "   训练参数:"
echo "     Phase 1: $PHASE1_STEPS steps, LR=$PHASE1_LR, warmup=$PHASE1_WARMUP"
echo "     Phase 2: $PHASE2_STEPS steps, LR=$PHASE2_LR, warmup=$PHASE2_WARMUP"
echo "     Batch: $BATCH_SIZE, Effective batch: $((BATCH_SIZE * 16))"
[[ -n "$LATEST_CKPT" ]] && echo "   断点续训: step $LATEST_STEP ($LATEST_CKPT)"
