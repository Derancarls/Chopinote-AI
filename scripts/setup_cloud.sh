#!/usr/bin/env bash
set -e

# ════════════════════════════════════════════════════════
# Chopinote-AI 云服务器一键部署脚本
# 用法:
#   bash scripts/setup_cloud.sh                    # 交互式
#   bash scripts/setup_cloud.sh --data-url <URL>   # 自动下载数据
# ════════════════════════════════════════════════════════

REPO_URL="https://github.com/Derancarls/Chopinote-AI.git"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()   { echo -e "${RED}[ERR]${NC}  $1"; }

# ── 解析参数 ──────────────────────────────────────────
DATA_URL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-url) DATA_URL="$2"; shift 2 ;;
        *) err "未知参数: $1"; exit 1 ;;
    esac
done

# ── 1. 工作目录 ───────────────────────────────────────
mkdir -p "$WORK_DIR" && cd "$WORK_DIR"
info "工作目录: $(pwd)"

# ── 2. 克隆 / 更新代码 ────────────────────────────────
if [ -d "Chopinote-AI/.git" ]; then
    info "仓库已存在，执行 git pull..."
    cd Chopinote-AI
    git pull
else
    info "克隆仓库..."
    rm -rf Chopinote-AI 2>/dev/null || true
    git clone "$REPO_URL"
    cd Chopinote-AI
fi

# ── 3. 安装依赖 ───────────────────────────────────────
info "安装 Python 依赖..."
pip install -e . --quiet

info "依赖安装完成！"
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python -c "import music21; print(f'  music21 {music21.__version__}')"

# ── 4. 数据 ──────────────────────────────────────────
if [ -d "data/processed" ] && [ "$(ls -A data/processed 2>/dev/null)" ]; then
    info "数据已存在: data/ (现有文件已保留)"
elif [ -n "$DATA_URL" ]; then
    info "从 $DATA_URL 下载数据..."
    wget -O data.tar.gz "$DATA_URL"
    tar -xzf data.tar.gz
    rm data.tar.gz
    info "数据解压完成"
else
    warn "未检测到 data/，也未提供 --data-url"
    echo ""
    echo "  请将 data/ 目录上传到 $(pwd)/data/"
    echo "  或重新运行: bash scripts/setup_cloud.sh --data-url <下载链接>"
    echo ""
fi

# ── 5. 输出汇总 ─────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  部署完成！"
echo ""
echo "  启动训练:"
echo "    cd $(pwd) && python scripts/run_training.py"
echo ""
echo "  查看日志:"
echo "    tail -f logs/train.log"
echo "═══════════════════════════════════════════════════"
