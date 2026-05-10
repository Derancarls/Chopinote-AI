# Chopinote-AI Roadmap

> 已实现功能按版本号记录。设计思路和发散讨论见 `design_docs/`。
> 格式：`vX.Y.Z-阶段` — Y 增量为大版本（架构/能力升级），Z 增量为小版本（优化/修复）。

---

## 初始阶段（未标记版本）

### 初始提交
- 项目仓库初始化
- 数据集处理模块和配置文件

### Beta1.0 — 基础模型管线
- **REMI Tokenizer**：175 词表，支持 Bar/Position/Track/Note_ON/Velocity/Duration
- **MusicXML→REMI Converter**：解析双轨钢琴谱，16 分格对齐，左右手分离
- **MusicTransformer**：6 层 Decoder-only，d_model=256，8 头，~5.3M 参数
- **Memory-mapped Dataset**：流式加载，适配 8GB RAM 场景
- **训练循环**：gradient accumulation，linear warmup + cosine decay
- **推理生成**：top-k 采样 + KV cache，支持导出 MusicXML
- **数据准备**：music21 内置语料库筛选 + 本地 MusicXML 支持
- 适配 GTX 750 Ti (4GB VRAM)：batch_size=2，max_seq_len=2048

---

## v0.1.0 系列 — 基础能力建设

### v0.1.0-beta1 — 首个可运行版本
- 基础模型 + 生成管线可用

### v0.1.0-Beta5 — 乐谱标记系统
- **10 种 token 类型扩展**：Clef/Dynamic/Hairpin/Artic/Ornament/Pedal/Slur/Repeat/Jump/Tempo
- **生成标记支持**：力度/演奏法/装饰音/踏板/渐强，XML 后处理注入
- **CLI 命令行工具**：`pip install -e .` 可安装，端到端续写流程
- **pyproject.toml**：项目打包配置

### 修复批次（v0.1.0 系列）
- **因果掩码修复**：之前所有位置可看到未来 token，改为完整下三角掩码
- **train.py 三项 bug 修复**：torch.load 兼容性、loss 累计错误（虚低 4 倍）、scheduler 步数偏差
- **扩展词表至 236**：新增 Clef/Dynamic/Hairpin/Artic/Ornament/Pedal/Slur/Repeat/Jump/Tempo
- **notes_to_score 时值 clamp**：防止 music21 makeNotation 崩溃
- **Diminuendo 丢失修复**
- **MetronomeMark 崩溃修复**
- **Weight tying 梯度正确性修正**
- **Loss NaN 修复**
- **KV cache 正确性修正**
- **无用 import 清理**

---

## v0.1.1 系列 — 功能完善

### v0.1.1-Beta4 — 连音与拍号支持
（对应 commit `bcc08149`，词表 236→271）
- **Tuplet 连音**：TupletStart/TupletEnd token，converter 检测，生成端 tuplet 时间缩放与 DurationTuplet 导出
- **TimeSig 拍号**：新增 33 个 TimeSig token，MusicXML/PDMX 自动提取拍号
- **converter priority 排序**：按事件类型优先级排序，确保 Position→Program→Note_ON 等顺序正确

### 云环境适配（CloudTrain 分支合并）
- **云服务器一键部署脚本**：setup_cloud.sh，含 CUDA/FFmpeg/项目依赖自动安装
- **data.tar.gz 自动解压**、PDMX.tar.gz 自动解压
- **PDMX 预处理管道修复**：CLI 可覆盖 cfg 模板参数
- **全局一致性修复**：Position 钳位、generate.py 死代码清理、PDMXToREMI API 一致化
- **_find_pdmx_files 正确剪枝 metadata 目录**：防止 254K 元数据 JSON 被误处理
- 数据加载优化、预处理阈值放宽

### v0.1.1-Beta3 — 预设系统与多轨完善
（对应 commit `d9833069`）
- **预设模板系统**：7 种预设（baroque/romantic/classical/dense/simple/minimal/jazz）
- **条件注入**：--condition-key / --condition-time / --condition-tempo
- **移调数据增强**：--augment-transpose
- **生成后自动验证**：--validate
- **CLI 全参数模式自动保存**：跳过交互，适合批处理
- **多轨乐器信息全流程保持**：converter.py part_program_map，Program token 始终保留
- **max_seq_len 超限优雅停止**
- **重命名 chopinote-generate → chopin**
- **Windows GBK 编码兼容**
- **test seed 生成器**、验证脚本

### FlashAttention + fp16 + 调性标记 — 重大更新
（对应 commit `02a58a0a`，词表 815）
- **FlashAttention**：`F.scaled_dot_product_attention` 替代手动注意力，自动选择最优后端
- **Gradient checkpointing**：减少训练显存占用
- **fp16 混合精度训练**：autocast + GradScaler
- **相对位置注意力**：可学习 `rel_bias` 参数，Music Transformer 方案
- **30 个 Key 调性标记 token**：词表 785→815，MusicXML/PDMX 自动提取调性
- **音高限制**：GM_INSTRUMENT_RANGES，按乐器范围屏蔽非法音高
- **多项修复**：compute_loss ignore_index、evaluate autocast、GradScaler 持久化、废弃代码清理

### 标记系统 + 乐理验证 + 4 轨种子
（对应 commit `90e450aa`）
- **generate.py/CLI 标记支持**：力度/演奏法/装饰音/踏板/渐强标记，XML 后处理注入
- **CLI 全功能链**：预设系统、多轨锁定、复杂度 auto-adjust、复音上限、调性偏置
- **validate_generation.py**：乐谱乐理合法性验证（同音重复/时值溢出/零时值/连音配对/空小节）
- **create_4track_seed.py**：钢琴双轨 + 小提琴双轨 + 力度踏板的 4 小节种子生成
- **设计文档**：modification_directions.md（记录乐器分轨、和声理解已知问题）

### v0.1.1-Beta5 — 移调增强与监控
（对应 commit `6ffc739b`）
- **PDMX 离线移调增强**：PDMXPreprocessor + CLI --augment-transpose/--transpose-range
- **TensorBoard 监控**：train/loss、train/lr、train/grad_norm、val/loss
- **Octave/Arpeggio 标记**：词表 831→837
- **调性偏置 bug 修复**：`'<Key '` → `'<Key'` 空格解析错误导致 key name 始终为空
- **max_polyphony 默认值统一**：10（函数签名/生成调用/CLI 注入三处对齐）
- **词表默认值对齐**：ModelConfig.vocab_size 831→837
- **死代码清理**：compute_loss、_identify_hands、TokenDataset 路径去重

### 乐器分轨改进
- **乐器级复音上限**：`INSTRUMENT_POLYPHONY_CAP`（弦乐/铜管/木管 ≤2，钢琴 ≤10，其他 4-8）
- **Per-track polyphony 追踪**：按 (program, subtrack) 独立追踪
- **弦乐 subtrack 级音域**：Violin/Viola/Cello/Contrabass/Tremolo/Pizzicato/Ensemble 划分

---

## v0.1.2 系列 — 模型升级与训练体系

### v0.1.2-Beta1 — 模型升级与训练管线
（对应 commit `e357fb38`）
- **模型架构升级**：d_model=768→1024，n_layers=10→12，n_heads=12→16，d_ff=3072→4096（74M→156M params）
- **MIDI→REMI 转换管道**：MIDIToREMI + MIDIPreprocessor + prepare_corpus --include-midi
- **分层训练**：TokenLossMask / PhaseConfig + _train_multiphase + 训练入口脚本
- **PDMXToREMI 导出修复**：统一导出接口
- **完整测试体系**：13 个测试文件，87 个单元测试
- **设计文档**：reference_models_research.md（音乐生成模型调研）
- **LICENSE 添加**

---

## 当前状态（v0.1.2-Beta1 之后）

### 配置与文档更新
- README 重构：更清晰的项目介绍和快速开始指南
- CLAUDE.md 同步更新
- .gitignore 重构：移除 Claude 配置文件的 git 追踪

### 数据下载（最新）
- 新增 POP909 数据集（909 首，已下载待解压）
- 新增 EMOPIA 数据集（1,087 首，已下载待解压）
- 新增 ATEPP 数据集（144 首，已下载待解压）
- 新增 MusicNet 数据集（330 首，已下载已解压）

### 项目目录重组
- 生成产物统一移至 `data/outputs/`
- 模型权重统一移至 `checkpoints/`
- 归档文件统一移至 `archives/`
- 笔记移至 `design_docs/`
- 移除重复数据集 `giant-midi-repo-2`

---

## 未实现 / 待定（见 design_docs/modification_directions.md）

> 设计思路和发散讨论完整记录在 `design_docs/` 目录下。
> 以下链接指向各设计文档，只有正式决定加入的内容才会写入本 Roadmap。

- [相对音高编码（音程制）](design_docs/tonality_and_harmony_scheme.md)（Tier 3 变体）
- [Chord 功能和弦标记](design_docs/tonality_and_harmony_scheme.md)（Tier 2）
- [Scale Degree 相对音级编码](design_docs/tonality_and_harmony_scheme.md)（Tier 3）
- [Anticipatory 控制点](design_docs/reference_models_research.md)
- [Compound Word + Delay Pattern](design_docs/reference_models_research.md)
- [FC-Attention 超长序列](design_docs/reference_models_research.md)
- [Cadenza 两层分离架构](design_docs/reference_models_research.md)
- [不限轨方案 B：Program + Voice](design_docs/unlimited_tracks_B_voice_scheme.md)
- [条件嵌入层（路径 B）](design_docs/conditional_embedding_path_b.md)
