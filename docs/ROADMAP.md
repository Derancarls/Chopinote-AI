# Chopinote-AI Roadmap

> 已实现功能按版本号记录。Y 增量为大版本（架构/能力升级），Z 增量为小版本（优化/修复）。
> 每个版本对应的已知问题见 `known_issues_<version>.md`。
>
> **当前训练状态**：Phase 1 预训练中（2026-05-27，step 12000 恢复，当前 step ~21500/140000，bs=8, grad_accum=4, effective_bs=32）
> loss ~0.73, NaN ~3%, ~9 天预估 → Phase 2 微调 50k steps
> 上次重启原因：n_section_bars_classes 128→2048 导致 bars_head 损坏，已回退 128

---

## 初始阶段（未标记版本）

### 初始提交
- 项目仓库初始化
- 数据集处理模块和配置文件

### Beta1.0 — 基础模型管线
- **REMI Tokenizer**：175 词表，Bar/Position/Track/Note_ON/Velocity/Duration
- **MusicXML→REMI Converter**：解析双轨钢琴谱，16 分格对齐，左右手分离
- **MusicTransformer**：6 层 Decoder-only，d_model=256，8 头，~5.3M 参数
- Memory-mapped Dataset 流式加载
- 训练循环：gradient accumulation + linear warmup + cosine decay
- 推理生成：top-k + KV cache + MusicXML 导出

---

## v0.1.0 系列 — 基础能力建设

### v0.1.0-Beta5 — 乐谱标记系统
- 10 种 token 类型扩展：Clef/Dynamic/Hairpin/Artic/Ornament/Pedal/Slur/Repeat/Jump/Tempo
- CLI 命令行工具：`chopin` 命令入口
- pyproject.toml 打包配置

### 修复批次
- 因果掩码修复、train.py 三项 bug、Loss NaN、KV cache 正确性等

---

## v0.1.1 系列 — 功能完善

### v0.1.1-Beta4 — 连音与拍号支持
- Tuplet 连音、TimeSig 拍号、converter priority 排序
- 词表 236→271

### 云环境适配（CloudTrain 分支合并）
- setup_cloud.sh 一键部署、PDMX 预处理管道修复

### v0.1.1-Beta3 — 预设系统与多轨
- 7 种预设模板、移调增强、多轨保留
- `chopin` 重命名、Windows GBK 兼容

### FlashAttention + fp16 + 调性标记
- `F.scaled_dot_product_attention` 替代手动注意力
- Gradient checkpointing、fp16 混合精度
- 30 个 Key 调性标记，词表 785→815

### v0.1.1-Beta5 — 移调增强与监控
- PDMX 离线移调增强、TensorBoard 监控
- Octave/Arpeggio 标记，词表 831→837
- 乐器级复音上限、subtrack 级音域

---

## v0.1.2 系列 — 模型升级与训练体系

### v0.1.2-Beta1 — 模型升级
- d_model=768→1024，n_layers=10→12，n_heads=12→16（74M→156M）
- MIDI→REMI 转换管道、分层训练、13 个测试文件 87 个单元测试

### v0.1.2-alibi — ALiBi 位置编码
- 可学习 position bias → ALiBi（相对位置编码，无需学习参数）
- FP8Linear（可选），词表 815→831

### v0.1.2-rope — RoPE 替代 ALiBi
- **RoPE 旋转位置编码**，SDPA is_causal=True 快速路径
- 4.1x 加速（Flash Attention 内的 RoPE 比 ALiBi 手工注意力快得多）

### v0.1.2-rope-measure — 恢复小节位置感知
- `measure_embedding`：nn.Embedding 映射小节号→d_model
- 与 RoPE 互补（RoPE 编码 token 位置，measure_embedding 编码节拍层级）

### v0.1.2-rope1 — 训练体系完善
- 训练质量指标（per-token-type accuracy）
- 数据增强、Launch Control 点火控制台（preflight → launch → monitor → abort）
- 词表 837→872

### v0.1.2-rope2 — 目录整理与优化
- scripts 分目录、bs8 优化、数据预处理管道完善

### 1.01B → 1.22B 扩参
- RTX 5090 OOM 修复、模型扩至 1.22B（后调为 1.21B）
- 多转换器 Bug 修复、训练功能增强
- Trainer.load_checkpoint 支持 vocab 扩展，自动复用旧 embedding

---

## v0.1.3 系列 — 评估与反馈

### v0.1.3 — Bug 修复
- MIDI 转换器修复、模型注意力掩码修复等 4 个 Bug

### v0.1.3-eval — 音乐性评价模块
- 两层评价：广义（音高密度/复音/音程分布/移调弹性）+ 狭义（旋律性/多声平衡/节奏密度弹性）
- 集成到 `chopin` CLI

### 修复批次
- 转调后音程主音更新、统一 `_infer_genre`、死代码清理

### v0.1.3-eval2 — 生成-评价反馈闭环
- **A/B1/B2/C 四阶段闭环**：生成 → 评价 → 退回重写 → 评价
- evaluator 反馈控制器、多模块修复

---

## v0.2.0 系列 — 渲染器与 DPO

### v0.2.1 — 渲染器 + 反馈训练
- **完整 REMI→MusicXML 渲染器**：13 维同度测试（v0.2.1）
- **DPO 偏好微调** + 退回重写机制
- 上下文传播机制

---

## v0.2.2 系列 — 项目结构整理

### v0.2.2 — 项目结构清理
- scripts 分目录（preprocess/train/generate/analysis）
- gitignore 清理（排除所有 XML/MIDI/生成文件）
- 设计文件全历史用 filter-branch 清除，docs/ gitignored 本地保留
- CLAUDE.md 重建

### v0.2.2-dev2 — tokenizer 扩展 + 渲染器重构
- tokenizer 微调、渲染器 bug 修复

---

## v0.2.3 系列 — 段落感知 + 和弦感知架构

### v0.2.3-dev1 — 段落感知全栈实现
**模型层**：
- `section_embedding` + `section_type_embedding`（实例级 + 类型级双层表征）
- `sec_bias` 四源可学习偏置（α 同实例凝聚 / β 同类型跨实例弱连接 / γ 跨类型分离 / δ 边界桥接），带 bar 距离指数衰减
- `SectionPredictionHead` 三头（bars 65 类 / key 31 类 / type 23 类）
- 手动 attention 路径（sec_bias 叠加）+ 标准 SDPA 回退

**训练**：
- 双任务 loss（next_token + 0.1 × section_prediction）
- anti-NaN guard（无段落数据时安全跳过）
- vocab/section_id 越界断言

**数据集**：
- TokenDataset 自动加载 `.sec.json` 段落标注
- collate_fn 同步 padding 段落字段

**生成**：
- `generate_structure_plan()` 结构规划（约束词表只允许结构 token）
- `section_aware_generate()` 段落条件生成（KV cache + 外部段落追踪）
- `SECTION_PARAMS` 段落参数联动

**标注管道**：
- `scripts/structure_annotator.py`：6 信号融合（key/density/repeat/tempo/program/silence）+ Viterbi 解码
- Sonata / Theme / Fallback 类型推断

**词表**：886→908（+21 Section types + <SecSum>）

**已知问题**：[known_issues_v0.2.3-dev1.md](known_issues_v0.2.3-dev1.md)

---

### v0.2.3-dev2 — 段落感知 Bug 修复 + CLI 集成 + 测试
- 段落感知架构 Bug 修复
- CLI 集成与预设系统完善
- 测试覆盖补充

---

### v0.2.3-dev3 — 和弦感知全栈实现 + 数据质量修复
**核心能力**：Structure → Harmony → Notes 三层层级生成

**和弦词表**：18 个罗马数字功能 token + 3 个转位 token，vocab 908→929

**模型新增**：
- `chord_embedding` + `chord_inv_embedding`（token-positional 注入）
- `ChordPredictionHead`（func 17 类 / inv 5 类）
- `Chord-Aware Attention Bias`（γ 同和弦凝聚 / ε 切换桥接 / ζ 同功能组弱连接 + δ sec_bias 去重）
- `chord_group_map` 三组映射（Tonic=0 / Subdominant=1 / Dominant=2）

**训练**：
- 三任务 loss（next_token + sec_loss_weight × sec_pred + chord_loss_weight × chord_pred）
- chord_func_ids 自动从 `.chord.json` 加载，无标注时回退全零
- Chord func/inv head 条件激活（仅在对应 token 位置贡献 loss）
- `load_checkpoint` embedding 行拷贝（vocab 扩展兼容）

**数据集**：
- TokenDataset 自动加载 `.chord.json` 和弦标注
- 4 个新字段（chord_func_ids / chord_inv_ids / chord_func_targets / chord_inv_targets）
- collate_fn 同步 padding

**生成**：
- `generate_harmony_skeleton()` 和声骨架（Chord→Chord7→Inv 状态机约束顺序）
- 三阶段生成：Stage 1 结构规划 → Stage 2 和声骨架 → Stage 3 条件音符填充（chord_bias 激活）
- `SECTION_HARMONY_PARAMS` 13 段落类型 × 和声密度/终止强度联动

**评价**：4 个新维度（和弦-旋律一致性 / 进行合理性 / 终止式质量 / 和声节奏），weight=0.06/0.06/0.04/0.04

**数据质量修复**：
- Key 追踪优先级反转：`chord_annotator` 始终从 `<Key>` token 序列追踪调性，`sec.json` 仅在第一 key 出现前 fallback
- `sec_keys_target`：`dataset.py` 改用 token-stream 精确 key_id，取代 `sec.json` 的段落聚合值（`Counter.most_common`）
- 侧边文件长度校验：`.sec.json`/`.chord.json` 加载时断言与 tokens 等长，防止过期数据静默错位
- 段落小节数上限 64→128（`config.n_section_bars_classes`）

**标注管线**：
- `scripts/chord_annotator.py`：模板匹配 + key 上下文 + 置信度 > 0.8 + 稀疏织体退化
- `scripts/parallel_chord_annotate.py`：22 核并行标注 + 调性覆盖统计报告（`_key_coverage_stats`）

详见：[functional_harmony_plan.md](functional_harmony_plan.md)

---

### v0.2.3-dev4 — SDPA 4D mask + 异步保存 + 标注修复 + fast_converter 上下文注入

**显存与性能优化**：
- `model.py`：段落注意力手动 softmax → SDPA 4D mask（避免 cuDNN math backend OOM）
- 限定 `_SDPA_BACKENDS_4D = [FLASH_ATTENTION, EFFICIENT_ATTENTION]`（排除 CUDNN）
- `train.py`：`save_checkpoint` 异步线程化，GPU→CPU 拷贝不阻塞训练
- loss reduction sum → mean（稳定训练曲线）

**监控增强**：
- evaluate 新增 sec/chord accuracy（per-token-type 指标）
- 验证集 accuracy 覆盖：overall/note/duration/bar/dynamic/velocity/key/tempo/sec_bars/sec_keys/sec_types/chord_func/chord_inv

**数据与标注修复**：
- `fast_converter.py`：每小节强制注入 Key/Tempo/TimeSig（确保采样窗口命中上下文）
- `dataset.py`：tokens_v2→v3；FIFO→LRU 缓存；num_workers=2, pin_memory=True
- `structure_annotator.py`：无段落时写整曲单段落而非空标注；BOUNDARY_THRESHOLD 0.4→0.2
- `chord_annotator.py`：`_key_coverage_stats` 改为每小节独立检查
- `launch_control.py`：适配新日志格式；PreflightChecker 指向 tokens_v3
- `requirements.txt`：添加 mido, tensorboard
- 测试适配：vocab_size 908→929, use_chord_attention=False, bars shape 动态化

---

### v0.2.3-dev5 — bias 重算入 checkpoint + detach，节省 8 GiB VRAM

**根因**：SDPA backward 在 `attn_mask` 需要梯度时会物化完整 `(B,nH,T,T)` = 8 GiB 注意力矩阵

**修复**：
- sec_bias/chord_bias 在 `_forward` 内从原始 `(B,T)` 数据（~1 MiB）重算，替代保存 `(B,1,T,T)` = 256 MiB → 节省 6 GiB
- 合并后的 bias 在传入 SDPA 前 `.detach()` → 避免 8 GiB mask 梯度物化
- `weakref.ref(self)` 实现偏置重算而不产生 nn.Module 注册循环
- 7 个可学习偏置标量（α/β/γ/δ/γ_chord/ε/ζ）保持合理默认值；section/chord embedding + prediction heads 提供主要学习信号

**效果**：bs=8 训练确认可运行（之前 OOM），配合 bs=12 仍需验证

---

### v0.2.3-dev6 — eval torch.no_grad() + DataLoader 修复 + README 双语重写

**稳定性修复**：
- `evaluate()` 添加 `torch.no_grad()` 防止验证时 autograd graph 累积，VRAM 从 31.2 GiB 降至安全水平，eval 加速 ~3x
- `val_dataloader` pin_memory=False（阻止 DataLoader worker 崩溃级联）
- CE loss NaN guard + fp32 转换防止 bf16 下溢
- `max_eval_batches` 500→200，eval 时间控制在 ~10 分钟

**文档**：
- README.md 重写为中英双语版，含详细架构分析、模型优势、设计推理

**已知问题继承自 dev1**（仍未修复）：
- P2：手动 attention 性能损失（15-22x）— 需 PyTorch 原生 custom bias SDPA 支持
- P3：标注质量未经评估
- P4：训练效果验证中（当前训练从零开始）

---

---

## v0.2.4 系列 — 训练稳定性 + CLI 配置系统

### v0.2.4-train1 — 训练稳定性与显存优化（2026-05-25）

**显存修复（31GB→22.3GB 稳定）**：
- `run_curriculum_training.py`：强制 `PYTORCH_CUDA_ALLOC_CONF`（`expandable_segments:True,roundup_power2_divisions:16,garbage_collection_threshold:0.6`）+ `set_per_process_memory_fraction(0.85)` 限制显存上限
- 模型直接 `model.bfloat16().to(device)` 创建，避免 fp32→bf16 中间态翻倍
- 训练前 `gc.collect()` + `torch.cuda.empty_cache()` 整理碎片
- `launch_control.py` watchdog 继承新 alloc conf；`--resume` 参数支持显式指定 checkpoint

**Step 计数器修复**：
- 恢复训练后显示 `Step 2,010/122,000` 而非 `Step 80/120,000`（`resume_offset = global_step` 偏移）

**推理 SDPA 鲁棒性**：
- `model.py`：Blackwell 某些长度下 SDPA 无可用内核时自动回退到手动 attention
- 推理时 `is_causal=True` 稳定（单 token 生成等价）

**训练质量修复**：
- `train.py`：optimizer param group 超参保留（防止旧 checkpoint 恢复时 KeyError）
- eval 阶段 `torch.cuda.empty_cache()` 释放校验缓存
- `harmony.py`：`pcs` 变量名覆盖内置 `sorted()` 修复
- `parser.py`：`getDuration()` → `duration.quarterLength`

---

### v0.2.4-config1 — CLI 配置文件系统（2026-05-26）

**文件结构**：
- `chopinote_cli/generation_config.yaml`：**唯一配置文件**，20 个生成参数，每项带取值范围 + 默认值 + 调高/调低效果说明
- `chopinote_cli/config.py`：`Config` dataclass + `load_config()` + `validate()` + `find_config()`
- 修改 `chopinote_cli/main.py`：集成配置加载与合并

**优先级链**：`CLI 参数 > --preset > 配置文件 > 内置默认值`

**核心设计**：
- 配置文件值为默认值，用户编辑 YAML 即控制所有生成变量
- 内置 `generation_config.yaml` 作为包默认配置，用户可创建 `./chopinote_config.yaml` 或 `~/.chopinote/config.yaml` 覆盖
- 无配置文件时保留原始交互式询问行为
- 配置校验：类型检查 + 范围越界检查 + 目标调性/曲式有效性

**CLI 新增参数**：
- `--config CONFIG`：指定配置文件路径
- `--random-seed`：自动生成随机种子实现可复现

**使用方式**：
```bash
chopin best.pt input.musicxml                          # 自动加载配置
chopin best.pt input.musicxml --config my_cfg.yaml     # 指定配置文件
chopin best.pt input.musicxml --temp 1.5 --max-bars 64 # 配置文件 + CLI 覆盖
chopin best.pt input.musicxml --random-seed            # 随机种子
```

---

### v0.2.4-dev1 — GPU 自动配置 + 推理优化 + 12 Bug 修复（2026-05-26）

**GPU 自动配置**（`chopinote_model/auto_config.py`）：
- 硬件检测：CPU 核心数、GPU 名称/VRAM/计算能力
- 自动推理配置：dtype (FP32/BF16/FP16/FP8)、torch.compile、TF32、线程数
- 训练提示：suggested_batch_size、suggested_fp8、suggested_gradient_checkpointing、suggested_memory_fraction
- `ModelConfig.from_gpu()` classmethod 自动调优
- 自打印硬件报告

**推理优化**（`main.py`）：
- `load_model()` 自动应用：dtype 转换、TF32、memory_fraction、FP8、torch.compile
- 线程数在首次 torch op 前设置（`torch.set_num_threads`）

**Bug 修复**（代码审查，不影响训练）：
- `has_cli_params` 在 config 覆盖前计算，避免误判
- seed 双重偏移修复（`generate_once` 已自行 offset）
- Optional type 校验兼容 `typing.Union` / `types.UnionType`
- Temperature 除零保护（3 处 `max(temp, 1e-8)`）
- 3 个 `open()` 添加 `encoding='utf-8'`
- 重复 `_prog_mode_bool` 删除
- `FP8Linear` 非 FP8 路径移除浪费的 `_update_scales()`
- CPU 核心数优先用 `psutil.cpu_count(logical=False)` 
- KV cache 空列表守卫（2 处）
- RoPE cache dtype 检查
- `--seed`/`--random-seed` argparse 互斥组

**新增 CLI 参数**：
- `--lock-program`、`--rest-penalty`、`--max-polyphony`
- `--key-bias`、`--prog-switch-strength`、`--prog-switch-interval`
- `--section-aware`、`--section-form`、`--section-total-bars`

---

### v0.2.4-train2 — Renderer 和弦修复 + 反馈空小节检测 + 训练稳定性（2026-05-27，HEAD）

**Renderer 和弦累积修复**（`renderer.py`）：
- `pending_interval` 单值 → `pending_notes: list[dict]` 列表累积
- Duration 到来时将累积的所有 Note_ON 写出（支持同位置和弦）
- 修复后音符丢失率从 91% (24/272) → 54% (124/272，剩余因模型语法不完整)
- Rest/Velocity/Artic/Ornament 同步适配列表模式

**反馈控制器空小节检测**（`feedback_controller.py`）：
- `empty_measure` 加入 B1_ADJUSTMENT_RULES（`rest_penalty +2.0, temperature +0.15`）
- `empty_measure` 加入 B2_ADJUSTMENT_RULES（`rest_penalty +2.5, temperature +0.2, complexity +1.5`）
- B1/B2 `_compute_b1`/`_compute_b2` 在 `window_notes < 3` 时仍运行 `_empty_measure_tokens`，不再跳过空窗口

**训练稳定性**：
- `n_section_bars_classes` 2048→128（2049 个输出神经元 94% 随机初始化，bars_head 无法学习）
- `dataset.py` bars_val clamp 2048→128
- `train.py` NaN 后 `total_loss = 0.0` 防止污染后续 logging 累加器
- sec_bars 从 section loss 中移除，仅保留 keys+types

**推理**：
- CPU 推理支持（`CUDA_VISIBLE_DEVICES=""`）避免 GPU 争用
- torch.compile CUDA Graph 与 RoPE cache 冲突 → CPU 推理绕过

---

### 当前训练（2026-05-27 运行中）

- **模型**：1.21B, 24L/32H/2048d, RoPE, 段落感知 + 功能和声
- **GPU**：NVIDIA RTX 5090 32GB
- **配置**：bs=8, grad_accum=4 (effective_bs=32), bf16 AMP, gradient checkpointing, VRAM ~22.3 GiB
- **进度**：Phase 1 预训练 step ~21500/140000（15.3%），从 step_12000.pt 恢复，当前 loss ~0.73
- **计划**：Phase 1 共 140k steps → Phase 2 MusicXML 微调 50k steps
- **监控**：`tmux attach -t chopinote:0` / TensorBoard port 6006
- **生成质量**：step 12000 模型严重欠训练（loss 0.94），生成 REMI 语法有错（48% Note_ON 无 Duration），需等 loss 降至 0.3-0.5 才稳定

---

## 规划中

### 待排期

| 方向 | 优先级 | 说明 | 依赖 |
|------|--------|------|------|
| **训练完成 → 效果评估** | **P0** | Phase 1 预训练完成后全面评估（loss 曲线、per-token-type accuracy、生成样本人工听评）。决定 Phase 2 微调是否继续或调整超参 | 当前训练 |
| **P2: 手动 attention 性能优化** | **高** | sec_bias 手动 attention 比 SDPA 慢 15-22x。等待 PyTorch 原生 custom attention bias API 或改 sdpa_kernel 后端。当前不影响训练（训练不开 sec_bias 手动路径），但生成推理速度受限于此 | torch.compile 兼容性 |
| **MuseScore 插件版** | 高 | 完整验证通过后，抽轻量版做 MuseScore 插件。用户可在 MuseScore 内一键调用模型续写/生成，实时渲染乐谱。需考虑：模型量化/蒸馏缩小体积、插件 API 封装（QML + 本地服务进程通信）、生成进度反馈 | 训练完成 + 效果达标 |
| ~~交互重构：CLI + 配置文件~~ | ✅ | **v0.2.4-config1 已完成**。YAML 配置文件 + Config dataclass + CLI `--config` 参数，20 项生成参数集中管理，优先级链 CLI > preset > config | 无 |
| **A 阶段增强：seed 分段、和弦标注与段落长度规划** | 中 | 三合一：(1) seed 段落边界检测，(2) seed 和弦功能标注，(3) **`section_length_planner()` 预计算各段长度**。取消模型 bars_head，段落长度不再作为模型预测目标，改由 A 阶段根据 seed 长度 + 段落类型分布函数算出初始长度（默认 4-20 小节，seed 越短弹性越大），B 阶段可动态微调。用户可通过 `section_length_multiplier` 参数整体缩放 | 训练基本可用 |
| **B1 段落边界感知** | 中 | 在段落分割位置强化反馈处理逻辑：边界前按前段落风格约束、边界后按后段落风格快速切换，避免段落过渡处的生成崩坏或风格混淆 | 训练基本可用 |
| **B2 段落级灵活漂移** | 中 | 允许不同段落之间存在风格差异，但约束同一段落的风格相对稳定。B2 评分窗口按段落边界划分，对跨段落对比引入段落类型加权 | 训练基本可用 |
| **C 阶段多轨优化** | 中 | 多轨生成场景下加入轨间指标：声部独立性、节奏互补性、音区避让、声部进行规则。退回重写时支持单轨回滚 | 训练基本可用 |
| 相对音高编码（音程制） | 中 | 替代绝对半音程 NOTE_ON，提升移调泛化 | 训练完成（破坏性变更） |
| 不限轨方案（Program + Voice） | 低 | 突破 4 subtrack 限制 | — |
| 条件嵌入层（路径 B） | 低 | 外部控制参数注入 Transformer | — |
| FC-Attention / 超长序列 | 低 | 突破 4096 max_seq_len | — |
| 序列并行 / 多卡训练 | 低 | 支持更大模型 | — |
| ~~GPU 自动适配~~ | ✅ | **v0.2.4-dev1 已完成**。`auto_config.py` 硬件检测 + 自动推理/训练配置 | 无 |
