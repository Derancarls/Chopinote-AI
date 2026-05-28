#!/usr/bin/env python3
"""禁用 CuDNN SDP 后端后启动 chopin 生成。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

# 禁用 CuDNN SDP（Blackwell 兼容性）
torch.backends.cuda.enable_cudnn_sdp(False)
print(f"[INFO] Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"[INFO] Mem Eff SDP: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"[INFO] CuDNN SDP: {torch.backends.cuda.cudnn_sdp_enabled()}")

import runpy
runpy.run_module('chopinote_cli.main', run_name='__main__')
