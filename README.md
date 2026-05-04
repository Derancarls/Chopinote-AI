# Chopinote-AI

> 自动生成古典钢琴曲谱的 AI 系统

Chopinote-AI 的目标是从 PDF 格式的乐谱出发，经过 MusicXML 转换和 REMI Tokenization，训练深度学习模型来自动续写和生成古典风格的钢琴曲谱。项目名取自 **Chopin**（肖邦）+ **Note**（音符）+ **AI**。

---

## 整体流程

```
PDF乐谱 → MusicXML → REMI Token序列 → AI模型 → 自动续写/生成
```

| 阶段 | 输入 | 输出 | 状态 |
|------|------|------|------|
| ① PDF 解析 | PDF 乐谱文件 | MusicXML | ❌ 未开始，需外部工具 |
| ② MusicXML→REMI | MusicXML | Token 序列 (JSON) | ⚠️ 框架搭好，核心逻辑待实现 |
| ③ 数据集划分 | Token 序列 | Train/Val/Test 切分 | ❌ 未实现 |
| ④ 模型训练 | Token 序列 | 模型权重 | ❌ 未开始 |
| ⑤ 自动生成/续写 | 种子 Token | 续写 Token→MusicXML | ❌ 未开始 |

---

## 环境检查

### 基础环境

| 项目 | 值 | 说明 |
|------|-----|------|
| Python | 3.14.3 | ✅ |
| 虚拟环境 | `music_ai_env/` | ✅ 已激活 |
| pip | 25.3 | ✅ |

### 核心依赖

| 包 | 版本 | 用途 | 状态 |
|----|------|------|------|
| music21 | 9.9.1 | MusicXML 解析与乐理分析 | ✅ |
| PyYAML | 6.0.3 | 配置管理 | ✅ |
| numpy | 2.4.3 | 数值计算 | ✅ |
| torch | 2.11.0 (CPU) | 深度学习框架 | ✅ |
| tqdm | 4.67.3 | 进度条 | ✅ |
| pandas | 3.0.1 | 数据处理 | ✅ |
| matplotlib | 3.10.8 | 可视化 | ✅ |

### 缺失项

| 缺失组件 | 说明 | 建议方案 |
|----------|------|----------|
| **原始乐谱数据** | `data/raw/` 下所有目录为空 | 从外部获取 Chopin/Bach/Beethoven 的 MusicXML |
| **PDF→MusicXML 工具** | music21 无法直接解析 PDF | 使用 [MuseScore](https://musescore.org) 的 OMR 功能，或 [Audiveris](https://github.com/Audiveris) 开源 OMR 引擎 |
| **converter.py 核心逻辑** | 当前为存根，仅返回 `[1,2,3,4,5]` | 需要实现 MusicXML→REMI 的实际转换 |
| **tokenizer.py 核心逻辑** | 当前为存根，仅返回 `[1,2,3]` | 需要实现 REMI Token 字典构建 |
| **splitter.py** | 空文件 | 需要按 config.yaml 配置实现数据集划分 |
| **requirements.txt** | 不存在 | 便于环境复现 |

### 待修复问题

1. ~~**目录命名拼写错误**：`chopinote_dateset/` → `chopinote_dataset/`~~ ✅ 已修复
2. ~~**模块命名拼写错误**：`dateset.py` → `dataset.py`~~ ✅ 已修复
3. **`.gitignore`** 缺少 `music_ai_env/` 虚拟环境目录
4. **Python 3.14** 过于前沿（2025 年发布），部分库可能存在兼容性问题，建议考虑 Python 3.11/3.12

---

## 项目结构

```
Chopinote-AI/
├── config.yaml                    # 全局配置文件
├── hello.py                       # music21 测试脚本
├── chopinote_dataset/             # 核心代码包
│   ├── converter.py               # MusicXML → REMI 转换器 (存根)
│   ├── tokenizer.py               # REMI 分词器 (存根)
│   ├── processor.py               # 预处理管线
│   ├── dataset.py                 # 数据集加载器
│   ├── splitter.py                # 数据集划分 (空)
│   └── data/
│       ├── raw/
│       │   └── chopin/            # 原始乐谱目录 (空)
│       ├── processed/
│       │   ├── tokens/            # 处理后 Token (空)
│       │   └── metadata/          # 元数据 (空)
│       └── cache/                 # MD5 缓存 (空)
├── test/                          # 测试目录 (空)
├── music_ai_env/                  # Python 虚拟环境
└── .gitignore
```

---

## 配置说明

核心配置在 `config.yaml` 中，主要包括：

- **数据源**：支持 Chopin（250 首）、Bach（371 首）、Beethoven（32 首）
- **REMI 参数**：grid_size=16, velocity_levels=8，支持力度/踏板/装饰音
- **序列长度**：最小 100 token，最大 2000 token
- **数据集划分**：80/10/10，按作曲家和体裁分层
- **质量检查**：重复检测、损坏检测、MusicXML 校验

---

## 快速开始

```bash
# 激活虚拟环境
source music_ai_env/Scripts/activate

# 验证环境
python hello.py

# 验证配置加载
python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"
```

---

## Todo

- [ ] 收集原始 MusicXML 数据集
- [ ] 实现 MusicXML → REMI 转换器
- [ ] 构建 REMI Token 字典
- [ ] 实现数据集划分
- [ ] 设计并训练音乐生成模型
- [ ] 实现自动续谱推理管线
- [ ] 实现 Token → MusicXML → 音频回放
