# Chopinote-AI

自动生成古典钢琴曲谱的 AI 系统。

---

## v0.1.0-Beta4 (2026-05-04)

### 新增
- CLI 命令行续写工具 `chopinote-generate`（交互式参数、KV cache、tqdm 进度条）
- `pyproject.toml`：`pip install -e .` 可一键安装依赖
- 乐谱标记词表扩展（175 → 236）：谱号、力度、渐强/渐弱、演奏法、装饰音、踏板、连奏线、反复、跳转、速度

### 修复
- `converter.py`: `Diminuendo` 渐弱记号未被捕获，训练数据丢失渐弱事件
- `converter.py`: `MetronomeMark.number is None` 时崩溃（文字型速度标记如 "Andante"）
- `model.py`: 因果掩码在 KV cache 推理时错误应用（改为仅首次 forward 应用）
- `train.py`: Weight tying 导致 Embedding 层梯度累加两次，等效学习率翻倍（Optimizer 参数去重）
- `train.py`: CrossEntropy `reduction='mean'` 全 mask 时 NaN（改为 `sum` + 安全除法，共 3 处）
- `train.py`: Scheduler 按 batch step 而非 optimizer step 调度，warmup 与 decay 提前结束
- `generate.py`: 独立 `generate()` 无 KV cache，每步重算全序列注意力 O(T²)
- `generate.py`: Session 状态泄漏（训练→评估切换后 `model.eval()` 未在后续训练步恢复）
- `generate.py`: 删除无用 import（`key as key21`, `bar`）

### 移除
- `generate.py` 独立 `generate()` 函数改为使用 KV cache，兼容旧调用接口

---

## v0.1.0-beta1 (2026-05-04)

### 新增
- REMI Tokenizer（grid_size=16, velocity_levels=8, 词表 175）
- MusicXML→REMI 转换器（双手谱按谱号自动分配左右手）
- 数据集划分模块（80/10/10）
- Decoder-only Transformer 模型（6层, d_model=256, 8头, ~5.3M 参数）
- Memory-mapped 流式数据集加载（适配 8GB 内存）
- 训练循环（AdamW + warmup + cosine decay + 梯度累积）
- 自回归生成管线（top-k + temperature 采样, KV cache）
- 随机权重 MusicXML 导出测试
- 数据预处理脚本（music21 语料库 → token 序列）
- CUDA 12.1 + Python 3.12 训练环境适配

### 修复
- `converter.py`: `.flat` → `.flatten()`（music21 废弃 API）
- `converter.py`: `from music21 import rest` → `note.Rest`
- `converter.py`: BAR 元数据统计改用 `sum(1 for ...)` 避免 None 异常
- `prepare_corpus.py`: Opus 多乐章对象拆分处理
- `prepare_corpus.py`: Opus 同名文件添加 `_{idx}` 后缀防覆盖
- `model.py`: KV cache NoneType 初始化检查
- `model.py`: `compute_loss` 移除 `@torch.no_grad()`（训练时无法反向传播）

### 移除
- `music_ai_env/`（Python 3.14 CPU 环境，已废弃）
- `structure.txt`
- 旧版存根代码（converter.py / tokenizer.py 占位逻辑）

---

## v0.0.1 (2026-04-30)

### 新增
- 项目基础结构搭建
- `config.yaml` 数据集配置
- `processor.py` 预处理管线框架
