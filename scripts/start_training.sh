#!/bin/bash
# 不用 set -e：ls 找 checkpoint 可能返回非零，手动处理
set -uo pipefail

# ──────────────────────────────────────────────────────────────
# Chopinote-AI 训练启动器
#   1. tmux 保活 — 退出 SSH 后继续运行
#   2. 看门狗自动重启 + 崩溃原因记录
#   3. 文件锁防止进程打架
#   4. TensorBoard 独立窗口
#   5. GPU 性能调优（不增加显存压力）
# ──────────────────────────────────────────────────────────────

SESSION_NAME="chopinote"
LOCK_FILE="/tmp/chopinote-train.lock"
PID_FILE="/tmp/chopinote-train.pid"

# 目录
PROJECT_DIR="${CHOPINOTE_PROJECT_DIR:-/root/Chopinote-AI}"
DATA_DIR="${CHOPINOTE_DATA_DIR:-/root/autodl-tmp/data/processed}"
CHECKPOINT_DIR="${CHOPINOTE_OUTPUT_DIR:-/root/autodl-tmp/chopinote/checkpoints}"
LOG_DIR="${CHOPINOTE_LOG_DIR:-/root/autodl-tmp/chopinote/logs}"
TB_DIR="${CHOPINOTE_TB_DIR:-/root/autodl-tmp/chopinote/tensorboard}"
TB_PORT=6006

mkdir -p "$LOG_DIR" "$TB_DIR" "$CHECKPOINT_DIR"

# ── 用法 ────────────────────────────────────────────────────
usage() {
    cat <<'HELP'
用法:
  ./scripts/start_training.sh             启动训练
  ./scripts/start_training.sh --status   查看状态
  ./scripts/start_training.sh --stop     停止训练
  ./scripts/start_training.sh --logs     查看实时日志
HELP
    exit 0
}

# ── 状态 ────────────────────────────────────────────────────
status() {
    local running=false
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        running=true
        echo "✅ 训练会话运行中 (tmux: $SESSION_NAME)"
    else
        echo "❌ 训练未运行"
        return
    fi

    echo "   附加训练窗口:   tmux attach -t $SESSION_NAME:0"
    echo "   附加 TensorBoard: tmux attach -t $SESSION_NAME:1"
    echo "   TensorBoard:    http://localhost:$TB_PORT"
    echo "   日志目录:       $LOG_DIR"

    if [[ -f "$LOG_DIR/watchdog.log" ]]; then
        echo ""
        echo "   最近看门狗日志:"
        tail -3 "$LOG_DIR/watchdog.log" | sed 's/^/     /'
    fi
    if [[ -f "$LOG_DIR/crashes.log" ]] && [[ -s "$LOG_DIR/crashes.log" ]]; then
        local crash_count
        crash_count=$(grep -c "Traceback" "$LOG_DIR/crashes.log" 2>/dev/null || echo 0)
        echo "   崩溃次数:       $crash_count (详情: $LOG_DIR/crashes.log)"
    fi
}

# ── 停止 ────────────────────────────────────────────────────
stop() {
    echo "正在停止训练..."
    tmux send-keys -t "$SESSION_NAME:0" C-c 2>/dev/null || true
    sleep 3
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    rm -f "$LOCK_FILE" "$PID_FILE"
    echo "已停止所有进程"
}

# ── 子命令 ─────────────────────────────────────────────────
case "${1:-}" in
    --help|-h) usage ;;
    --status)  status; exit 0 ;;
    --stop)    stop; exit 0 ;;
    --logs)    tail -f "$LOG_DIR"/watchdog.log "$LOG_DIR"/crashes.log 2>/dev/null || echo "日志文件尚不存在"; exit 0 ;;
    --*)       echo "未知选项: $1"; usage ;;
esac

# ── 单实例锁 ────────────────────────────────────────────────
if [[ -f "$LOCK_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "❌ 训练已在运行 (PID $OLD_PID)"
        echo "   查看状态: $0 --status"
        echo "   强制停止: $0 --stop"
        exit 1
    fi
    echo "⚠️  发现残留锁文件，已清除"
    rm -f "$LOCK_FILE" "$PID_FILE"
fi
echo $$ > "$PID_FILE"
trap 'rm -f "$LOCK_FILE" "$PID_FILE"' EXIT

# ── GPU 性能调优（不增加显存压力） ──────────────────────────
# TF32: 免费 8 倍 matmul 加速，无显存开销，精度无损
export TORCH_CUDNN_V8_API_ENABLED=1
# 减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 单 GPU 不需要 NCCL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# CPU 线程限制（避免与数据加载抢核）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# tokenizer 不并行
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"

# ── 训练参数 ────────────────────────────────────────────────
PHASE1_STEPS=120000
PHASE1_LR=1.5e-4
PHASE1_WARMUP=4000
PHASE2_STEPS=50000
PHASE2_LR=1e-4
PHASE2_WARMUP=2000
BATCH_SIZE=2

# 检查最新 checkpoint 用于断点续训
LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/step_*.pt 2>/dev/null | head -1)
RESUME_ARG=""
if [[ -n "$LATEST_CKPT" ]]; then
    LATEST_STEP=$(basename "$LATEST_CKPT" | sed 's/step_//;s/\.pt//')
    if [[ "$LATEST_STEP" -gt 0 ]] 2>/dev/null; then
        RESUME_ARG="--resume $LATEST_CKPT"
        echo "📌 发现已有 checkpoint (step $LATEST_STEP)，将断点续训"
    fi
fi

# ── 看门狗脚本 ──────────────────────────────────────────────
# 写入临时文件避免 tmux 引用问题
cat > /tmp/chopinote-watchdog.sh << 'WDEOF'
#!/bin/bash
set -uo pipefail

# 从环境变量读取
PROJECT_DIR="{{PROJECT_DIR}}"
DATA_DIR="{{DATA_DIR}}"
CHECKPOINT_DIR="{{CHECKPOINT_DIR}}"
LOG_DIR="{{LOG_DIR}}"
TB_DIR="{{TB_DIR}}"
PHASE1_STEPS={{PHASE1_STEPS}}
PHASE1_LR={{PHASE1_LR}}
PHASE1_WARMUP={{PHASE1_WARMUP}}
PHASE2_STEPS={{PHASE2_STEPS}}
PHASE2_LR={{PHASE2_LR}}
PHASE2_WARMUP={{PHASE2_WARMUP}}
BATCH_SIZE={{BATCH_SIZE}}
RESUME_ARG="{{RESUME_ARG}}"
CRASH_LOG="$LOG_DIR/crashes.log"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"

MAX_RESTARTS=5
RESTART_DELAY=30

cd "$PROJECT_DIR"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] ⚙️  看门狗启动 (最多重启 $MAX_RESTARTS 次)" >> "$WATCHDOG_LOG"

RESTART_COUNT=0
while true; do
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] 🚀 训练启动 (重启 #$RESTART_COUNT)" >> "$WATCHDOG_LOG"

    # CUDA graph 清理（每次重启重置缓存）
    python -c "
import torch
# TF32 已在 run_curriculum_training.py 中设置
torch.cuda.empty_cache()
print(f'CUDA 缓存已清理 | GPU: {torch.cuda.get_device_name(0)} | '
      f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
" 2>> "$CRASH_LOG"

    python scripts/run_curriculum_training.py \
        --midi-train-list "$DATA_DIR/train.txt" \
        --musicxml-train-list "$DATA_DIR/train.txt" \
        --val-list "$DATA_DIR/val.txt" \
        --data-dir "$DATA_DIR" \
        --phase1-steps $PHASE1_STEPS \
        --phase1-lr $PHASE1_LR \
        --phase1-warmup $PHASE1_WARMUP \
        --phase2-steps $PHASE2_STEPS \
        --phase2-lr $PHASE2_LR \
        --phase2-warmup $PHASE2_WARMUP \
        --batch-size $BATCH_SIZE \
        --output-dir "$CHECKPOINT_DIR" \
        --log-dir "$TB_DIR" \
        $RESUME_ARG \
        2>> "$CRASH_LOG"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] ✅ 训练正常完成" >> "$WATCHDOG_LOG"
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))

    # 记录崩溃详情
    {
        echo "═══════════════════════════════════════════════"
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] ❌ 训练崩溃 (exit=$EXIT_CODE, 重启 #$RESTART_COUNT)"
        echo "  GPU 状态:"
        nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/    /'
        echo "  系统内存:"
        free -h | grep Mem | sed 's/^/    /'
        echo "═══════════════════════════════════════════════"
    } >> "$CRASH_LOG"
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ❌ 训练崩溃 (exit=$EXIT_CODE, 重启 #$RESTART_COUNT)" >> "$WATCHDOG_LOG"

    if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] ⛔ 已达最大重启次数 ($MAX_RESTARTS)，停止" >> "$WATCHDOG_LOG"
        break
    fi

    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ⏳ 等待 ${RESTART_DELAY}s 后重启..." >> "$WATCHDOG_LOG"
    sleep $RESTART_DELAY
done
WDEOF

# 替换模板变量
sed -i \
    -e "s|{{PROJECT_DIR}}|$PROJECT_DIR|g" \
    -e "s|{{DATA_DIR}}|$DATA_DIR|g" \
    -e "s|{{CHECKPOINT_DIR}}|$CHECKPOINT_DIR|g" \
    -e "s|{{LOG_DIR}}|$LOG_DIR|g" \
    -e "s|{{TB_DIR}}|$TB_DIR|g" \
    -e "s|{{PHASE1_STEPS}}|$PHASE1_STEPS|g" \
    -e "s|{{PHASE1_LR}}|$PHASE1_LR|g" \
    -e "s|{{PHASE1_WARMUP}}|$PHASE1_WARMUP|g" \
    -e "s|{{PHASE2_STEPS}}|$PHASE2_STEPS|g" \
    -e "s|{{PHASE2_LR}}|$PHASE2_LR|g" \
    -e "s|{{PHASE2_WARMUP}}|$PHASE2_WARMUP|g" \
    -e "s|{{BATCH_SIZE}}|$BATCH_SIZE|g" \
    -e "s|{{RESUME_ARG}}|$RESUME_ARG|g" \
    /tmp/chopinote-watchdog.sh
chmod +x /tmp/chopinote-watchdog.sh

# ── 启动 tmux ──────────────────────────────────────────────
# 先清理旧会话
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
sleep 1

# 新建会话（不附加）
tmux new-session -d -s "$SESSION_NAME" -x 160 -y 50

# 窗口 0: 训练看门狗
tmux rename-window -t "$SESSION_NAME:0" "train"
tmux send-keys -t "$SESSION_NAME:0" \
    "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1" Enter
tmux send-keys -t "$SESSION_NAME:0" \
    "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter
tmux send-keys -t "$SESSION_NAME:0" \
    "export TORCH_CUDNN_V8_API_ENABLED=1 TOKENIZERS_PARALLELISM=false" Enter
tmux send-keys -t "$SESSION_NAME:0" \
    "cd $PROJECT_DIR" Enter
tmux send-keys -t "$SESSION_NAME:0" \
    "bash /tmp/chopinote-watchdog.sh" Enter

# 窗口 1: TensorBoard
tmux new-window -t "$SESSION_NAME" -n "tensorboard"
tmux send-keys -t "$SESSION_NAME:1" \
    "tensorboard --logdir $TB_DIR --port $TB_PORT --bind_all 2>&1 | tee $LOG_DIR/tensorboard.log" Enter

# 写锁文件
echo $$ > "$LOCK_FILE"

# ── 输出信息 ────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              🎵 Chopinote-AI 训练已启动                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  📊 TensorBoard:  http://localhost:$TB_PORT                    ║"
echo "║                                                              ║"
echo "║  tmux 控制:                                                  ║"
echo "║    附加训练窗口:   tmux attach -t $SESSION_NAME:0              ║"
echo "║    附加 TensorBoard: tmux attach -t $SESSION_NAME:1            ║"
echo "║    分离:           Ctrl+B, D                                 ║"
echo "║                                                              ║"
echo "║  日志:                                                       ║"
echo "║    看门狗:         tail -f $LOG_DIR/watchdog.log              ║"
echo "║    崩溃记录:       tail -f $LOG_DIR/crashes.log               ║"
echo "║    TF 日志:        tail -f $LOG_DIR/tensorboard.log           ║"
echo "║                                                              ║"
echo "║  命令:                                                       ║"
echo "║    查看状态:       $0 --status                                 ║"
echo "║    停止训练:       $0 --stop                                  ║"
echo "║    实时日志:       $0 --logs                                  ║"
echo "║                                                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  训练参数:                                                    ║"
echo "║  模型: 1.22B params | batch_size=$BATCH_SIZE | accum=16 | seq=4096  ║"
echo "║  Phase 1: ${PHASE1_STEPS} steps, LR=${PHASE1_LR}, warmup=${PHASE1_WARMUP}     ║"
echo "║  Phase 2: ${PHASE2_STEPS} steps, LR=${PHASE2_LR}, warmup=${PHASE2_WARMUP}     ║"
echo "║  GPU: TF32 ON | 数据加载: 主进程 | 不爆显存 ✓              ║"
[[ -n "$LATEST_CKPT" ]] && echo "║  断点续训: step $LATEST_STEP                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
