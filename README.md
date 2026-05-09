# Chopinote-AI

输入一首钢琴曲（MusicXML），让 AI 帮你续写下去，生成完整的古典风格乐谱。

---

## 它能做什么

给你一首曲子的开头片段，比如肖邦的前奏曲前几小节，Chopinote-AI 会用自回归的方式逐 token 续写，生成风格匹配的后续乐章。最终输出标准的 MusicXML 文件，可以在 MuseScore、Finale、Sibelius 等打谱软件中打开和编辑。

支持以下音乐元素的生成：

- **音符骨架**：音高、节奏、和声、多声部
- **力度**：ppp 到 fff，渐强/渐弱
- **演奏法**：跳音、重音、保持音、琶音、装饰音（颤音/回音/波音）
- **踏板**：延音踏板记号
- **连音**：三连音、五连音等
- **乐器**：支持 General MIDI 标准的所有乐器，多轨同时生成
- **拍号/调号/速度**：可锁定或自由变化

---

## 快速开始

### 安装

```bash
pip install -e .
```

### 下载预训练权重

```bash
# 推荐使用 step_94000_best.pt（v0.1.2-Beta1）
# 下载后放到项目根目录即可
```

### 基本使用

```bash
# 交互式续写：输入一首 MusicXML，AI 帮你写完
chopin step_94000_best.pt input.musicxml

# 指定输出文件
chopin step_94000_best.pt input.musicxml -o output.musicxml

# 一次生成多份变体
chopin step_94000_best.pt input.musicxml -n 5
```

### 控制生成风格

```bash
# 使用预设风格
chopin step_94000_best.pt input.musicxml --preset baroque
chopin step_94000_best.pt input.musicxml --preset romantic
chopin step_94000_best.pt input.musicxml --preset jazz

# 手动调节参数
chopin step_94000_best.pt input.musicxml --temp 1.2 --top-k 40 --complexity 7

# 锁定部分音乐要素
chopin step_94000_best.pt input.musicxml --key-mode lock     # 保持原调性
chopin step_94000_best.pt input.musicxml --time-mode lock    # 保持原拍号
chopin step_94000_best.pt input.musicxml --tempo-mode lock   # 保持原速度
```

### 种子截取

```bash
# 只使用前 4 小节作为种子
chopin step_94000_best.pt input.musicxml --seed-bars 4

# 指定种子文件
chopin step_94000_best.pt input.musicxml --seed input_seed.musicxml
```

---

## 可用预设

| 预设名 | 说明 | 复杂度 |
|--------|------|--------|
| `default` | 默认平衡模式 | 5 |
| `dense` | 密集织体，音符多 | 7 |
| `simple` | 简单织体，稀疏 | 2 |
| `baroque` | 巴洛克风格 | 6 |
| `romantic` | 浪漫主义风格 | 6 |
| `classical` | 古典主义风格 | 4 |
| `minimal` | 极简风格 | 2 |
| `jazz` | 爵士风格 | 6 |
| `church` | 圣咏风格 | 3 |

---

## 训练自己的模型

### 数据处理

```bash
# 准备训练数据（MusicXML → token 序列）
python scripts/prepare_corpus.py

# 包含本地文件和 MIDI 文件
python scripts/prepare_corpus.py --include-local --include-midi

# 带移调增强（±5 半音 = 11 倍扩充）
python scripts/prepare_corpus.py --augment-transpose
```

### 训练

```bash
# 分层训练：先用 MIDI 预训练音符骨架，再用 MusicXML 微调细节
python scripts/run_curriculum_training.py \
    --midi-train-list data/processed/midi_train.txt \
    --musicxml-train-list data/processed/train.txt \
    --phase1-steps 250000 \
    --phase2-steps 100000
```

训练时自动启用 TensorBoard 监控，可实时查看 loss、学习率、梯度变化。

---

## 技术概要

| 模块 | 说明 |
|------|------|
| **模型架构** | Decoder-only Transformer，156M 参数，12 层，d_model=1024 |
| **tokenizer** | REMI 编码，grid_size=16，velocity_levels=8，词表 837 |
| **注意力** | FlashAttention + KV cache 推理 + gradient checkpointing 训练 |
| **数据来源** | MusicXML 语料库 + PDMX 数据集 + MIDI 数据集（Lakh/Aria/Bread/MAESTRO） |
| **训练策略** | 两阶段分层训练：Phase 1 MIDI 预训练（屏蔽表现力 token），Phase 2 全量微调 |
| **硬件** | RTX 4090 24GB（当前配置完全利用可用显存） |

---

## 项目结构

```
├── chopinote_cli/main.py           CLI 入口
├── chopinote_model/
│   ├── config.py                   模型和训练配置
│   ├── model.py                    Transformer 模型
│   ├── train.py                    训练循环
│   ├── generate.py                 推理生成
│   └── dataset.py                  数据加载
├── chopinote_dataset/
│   ├── tokenizer.py                REMI 编解码（词表 837）
│   ├── converter.py                乐谱 → token 转换
│   ├── processor.py                批量预处理
│   └── splitter.py                 数据集划分
├── scripts/
│   ├── prepare_corpus.py           数据准备
│   ├── preprocess_pdmx.py          PDMX 预处理
│   ├── run_curriculum_training.py  分层训练入口
│   └── validate_generation.py      生成结果验证
├── design_docs/                    设计文档
└── tests/                          测试（87 个）
```

---

## 版本历史

- **v0.1.2-Beta1**: 模型升级 156M、MIDI 转换管道、分层训练、测试体系
- **v0.1.1-Beta5**: 移调增强、TensorBoard 监控、调性 Bug 修复
- **v0.1.1-Beta4**: 连音 token、拍号 token、词表扩展
- **v0.1.1-Beta3**: 预设系统、多轨保持、CLI 完善
- **v0.1.0-Beta5**: 乐谱标记系统（力度/演奏法/装饰音/踏板）
- **v0.1.0-beta1**: 首个可运行版本，基础模型 + 生成管线
