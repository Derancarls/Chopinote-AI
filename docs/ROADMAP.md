# Chopinote-AI Roadmap

> 已实现功能按版本号记录。Y 增量为大版本（架构/能力升级），Z 增量为小版本（优化/修复）。
> 每个版本对应的已知问题见 `known_issues_<version>.md`。
>
> **当前状态 (2026-06-02)**：v0.2.x 训练已放弃 (step ~51000/166000)。v0.3.x 全部设计方案完成 (11 篇)，代码部分实现 (词表/模型/SSF/Voice/Fig)，数据管线/时值饱和度/终止式感知/框架分离/乐句层/课程训练待实现。全设计一次到位后启动从头训练。

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

## v0.1.3 系列 — ABC Engine 前身

### v0.1.3 — Bug 修复
- MIDI 转换器修复、模型注意力掩码修复等 4 个 Bug

### v0.1.3-eval — 乐谱打分层
- C 阶段前身：两层评分（广义 + 狭义）
- 集成到 `chopin` CLI

### 修复批次
- 转调后音程主音更新、统一 `_infer_genre`、死代码清理

### v0.1.3-eval2 — ABC Engine 原型
- **A/B1/B2/C 四阶段闭环**：生成 → 评分 → 退回重写
- 反馈控制器、多模块修复

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

### v0.2.4-train2 — Renderer 和弦修复 + 反馈空小节检测 + 训练稳定性（2026-05-27）

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

### v0.2.4-eval — ABC Engine 默认启用 + A 阶段蓝图 + B 段落感知 + C 诊断（2026-05-28）

**ABC Engine 默认启用**：
- A(感知层) B(决策层) C(进化层) 不需要 `--feedback` 标志，默认对生成结果执行三阶段认知处理
- B1/B2 调整规则补入 `empty_measure` 等明显生成失败指标

**A 感知层 (Perceptor)**：
- `SectionStyleTarget.from_seed_section()` 从 seed 真实 token 统计推导风格目标
- `HarmonyContext` 新增 `seed_contour`，供 B2 melodic_contour 对比
- 段落检测优先集成 `structure_annotator` 的 6 信号 Viterbi 方法
- `_build_section_plan` 支持按比例缩放段落长度

**B 决策层 (Reasoner)，含 B1/B2 子阶段**：
- B1 局部流畅 + B2 全局漂移双轨道
- B2 段落容忍度（`SECTION_B2_TOLERANCE`），development/cadenza 段降低调整力度
- `_cadence_placement_check`：终止式位置合理性检查

**C 进化层 (Reflector)**：
- 结构化小节级诊断报告（`diagnose_bars()`）

**新指标**（`registry.py`）：
- `_max_polyphony_per_position_tokens`、`_token_type_kl_tokens`
- `harmonic_rhythm_score_tokens` 支持 B1 绝对 / B2 相对双模式
- `_parallel_fifths_tokens` 重建为声部引导算法
- `_melodic_contour_tokens` 支持 seed_contour 参考对比

**CLI**：eval light 模式不传 checkpoint，减少推理开销

---

### v0.2.4-eval2 — A 阶段段落检测增强 + B1/B2 新指标 + bars_head 移除（2026-05-28）

- `bars_head` 从 `SectionPredictionHead` 中完整移除（仅保留 key_head + type_head）
- `SectionStyleTarget.from_seed_section()` 统计密度/休止比例/复杂度/力度均值 → 温度映射
- `_detect_sections_annotator` 集成 structure_annotator 6 信号 Viterbi
- 平行五度检测重写为声部引导算法
- `_token_type_kl`、`_max_polyphony_per_position` 注册为正式指标
- B 调整乘入 `section_tolerance` 因子

---

---

## v0.2.5 系列 — 模型优化 + ABC Engine v2 设计

### v0.2.5-dev1 — Effective Batch 64 + DPO 自动微调 + 评估优化（2026-05-28）

- effective batch size 调至 64（DRAGON baseline）
- DPO 微调自动化脚本
- 生成 CLI 语法补全 + eval batch 增至 100

### v0.2.5-dev2 — QK-Norm + 嵌入增强 + 训练特性（2026-05-29）

**注意力质量**：
- **QK-Norm**：per-head RMSNorm 对 Q/K 归一化（防止 attention logit 方差失控）
- **per-head Q/K scaling**：每个注意力头独立的可学习温度（头特异性）
- **Attention logit soft-capping**：cap=50（manual fallback 路径，防止极端 logit）

**嵌入层扩展**：
- **voice_count_embedding**：同 Position 下第几个音（声部计数感知）
- **measure_in_section_embedding**：段落内相对小节位置（结构级位置感知）
- 两个新嵌入 zero-init 不干扰预训练权重

**训练增强**：
- **Z-loss**（weight=1e-4）：辅助 loss 约束 logit 幅度，抑制 NaN
- **EMA**（β=0.999）：平滑权重跟踪，checkpoint 持久化
- **dropout 阶梯调度**：0.15→0.10(step 47000)→0.08(step 80000)
- **FP8 Linear 默认启用**（warmup 500），节省显存
- **Token 级加权 CE loss**：chord ×0.5 / position ×2.0 / 重复 ×1.2

**训练**：从 step_47000.pt 恢复，bs=8, grad_accum=2, effective_bs=16

### v0.2.5 — 合并到 main，正式版（2026-05-30）

---

## v0.2.6 系列 — ABC Engine v2 Phase 1

### v0.2.6-abc1 — ABC Engine v2 Phase 1 规则实现（2026-05-30）

**新建 `chopinote_abc/` 包**（~1100 行，零模型依赖）：

**A1 框架记忆库**（`database.py`）：
- `SectionPlan` / `ChordAtBar` / `SeedContext` / `StructuralFix` dataclasses
- A1DB CRUD + prefix 序列化（structure_tokens / harmony_tokens）
- 运行时覆盖（overrides）+ C→A1 修复写回（apply_fix）
- 段索引（_reindex）+ 段落查找（find_section / get_section / get_chord）

**A2 动机摘要库**（`database.py` + `motif.py`）：
- `MotifDNA`（contour / rhythm / scale_degrees / strong_beat_mask / register）
- `MotifRecord` + A2DB CRUD + 相似度搜索
- `identify_landmarks()`：语义驱动地标 bar 选择（statement / climax / distinctive）
- `purify_tokens()`：剥离演奏层（8 种前缀），Velocity→4 归一化
- `extract_dna()`：从提纯 token 提取 MotifDNA

**A3 统计画像库**（`database.py`）：
- `BarStats` / `SectionStats`（density / pitch_range / velocity / rest_ratio / harmonic_rhythm / pitch_class_dist / interval_dist / token_type_counts）
- A3DB 追加式 bar_log + 段快照 + 基线（set_baseline）
- 统计查询：get_window / get_trend / get_cumulative / compare_sections / compare_to_baseline

**Stage 1/2 规则规划器**（`planner.py`）：
- `plan_structure()`：曲式模板（sonata/binary/theme_variations/free）→ 段落分配 + 调性规划
- `plan_harmony()`：14 段落类型和声模板 + 终止式约束 + 段间 pivot chord
- `reharmonize_from_bar()`：局部和声回退（B 触发）
- `tonal_progression_template()`：模板进行 + 终止式替换

**B 决策层**（`decision.py`）：
- `BHardBans`：7 类硬约束 token 屏蔽（平行五度/平行八度/声部交叉/音域超限/极端跳跃/导音未解决/contour 偏离）
- `apply_zone_temperature()`：段内冷→热→冷温区退火（1 bar 线性过渡）

**Stage 3 迭代生成**（`generate.py` 新增 ~200 行）：
- `stage3_iterative_generate()`：逐段循环生成 + max_retries C→A1 闭环
- `_stage3_generate_once()` + `_c_evaluate()` Phase 1 规则版

**设计文档**：`docs/abc_engine_inference_loop.md` v3（五层 prefix / KV cache 策略 / 创新预算 / Phase 1-3 分期规划）

---

### v0.2.6-abc2 — DPO 自动闭环 + C 增强 + 日志系统 + B1/B2 六层拆分（2026-05-31）

**DPO 自动优化闭环**（`train.py` + `batch_evaluate.py`）：
- `_run_eval_generation()`：训练中自动批量生成（seed×temp×samples）→ C 评分 → write_reward_log
- `_check_dpo_trigger()`：reward_log 新增 ≥ 20 条 → 自动拉起 DPO
- `_run_dpo_phase()`：build_preference_dataset → LoRA (QKV, rank=8) → DPO train 3 epoch → merge
- pre/post-DPO checkpoint 自动保存，所有 eval/DPO 功能默认关闭

**C 评价层增强**（`scoring.py`，+300 行）：
- **MusicXML 快速审查**：`review_musicxml()` → `BarInspection`（逐 bar 声部沉默/音域违规/声部交错/平行五八度/密度极端）
- **Token↔XML 对比**：`compare_tokens_to_xml()` → fidelity 保真度分数
- **C→B 结构化反馈**：`CFeedback` dataclass（ban_pitches/part_bias/temperature_delta/complexity_delta/fatal）
- 段间 `_apply_section_c_feedback()`：每段完成后实时调参

**ABC 日志系统**（`logging.py`，~600 行）：
- `ABCGenerationLogger`：每次生成独立日志文件 + JSON 汇总
- 六层全覆盖（A1/A2/A3/B1/B2/C），组件间数据流追踪
- 控制台 WARNING+ / 文件 DEBUG 全量

**B 层六层拆分**（B1 硬约束 + B2 决策调参）：
- `GenerationSummary` B1/B2 指标分离
- `decision.py` BHardBans/BFeedback 独立
- 9 处 call site 更新

---

### v0.2.6-abc3 — 渲染器重写 + 日志重构 + 禁令调优 + FP8 推理（2026-05-31）

**渲染器性能重写**（`renderer.py`）：
- `_inject_directions` + `_cleanup_accidentals` 改用 `xml.etree.ElementTree`（O(E×S)→O(S+E×logM)）
- 新增 `_render_raw_xml()` fast path：直写 MusicXML 绕过 music21（**0.01s vs 540s，54000x 加速**）
- `save_to_musicxml` 加 `export_midi` / `fast_path` 参数
- 修复旧 music21 `_build_score` 丢 92% 音符的 bug

**日志系统重构**（`logging.py`）：
- 双 Formatter：`ABCPlainFormatter`（文件纯文本，ESC=0）+ `ABCColorFormatter`（终端彩色）
- `AsyncFileHandler`：Queue + daemon 线程异步写，不阻塞生成主循环
- Token 级日志：每 token 采样详情（prob/T/topK/logit_range）、每 bar token 清单
- 前向传播耗时记录（`log_forward_pass`）

**禁令调优**：
- 上下文禁令 570→~150（仅 Tuplet/GraceNote/非seed Program）
- Tempo/TimeSig 恢复自由采样，模型能正常结束小节
- Fallback 从 (96,3) 收紧到 (48,6)
- 密度从 92→24.3 notes/bar（接近纯模型 19.8，seed 12.8）

**推理加速**：
- FP8 推理默认启用（`model.set_fp8_mode(True)`，`_scaled_mm` 加速）
- Lucy seed 命名（`lucy_seed_C_major_4bar` / `lucy_seed_Gb_major_4bar`）

**设计文档**：`docs/abc_engine_roles.md`（六层角色刻画）、`docs/bar_structural_plan.md`（Bar A1 预规划架构）

---

### 当前训练（2026-05-31 运行中）

- **模型**：1.21B, 24L/32H/2048d, RoPE, QK-Norm, 段落感知 + 功能和声 + 声部/节位嵌入
- **GPU**：NVIDIA RTX 4090 48GB
- **配置**：bs=8, grad_accum=2 (effective_bs=16), bf16 AMP, FP8, gradient checkpointing, VRAM ~22.7 GiB
- **进度**：Phase 1 预训练 step ~51000+/166000（~31%），从 step_50000.pt 恢复，当前 train loss ~0.67
- **速度**：~5.8s/step
- **计划**：Phase 1 共 166k steps → Phase 2 MusicXML 微调 50k steps
- **监控**：`python scripts/train/launch_control.py monitor` / TensorBoard

---

## 规划中

> 以下按依赖顺序排列。**所有设计在训练启动前一次实现到位**，不设中间版本训练。
> 当前阶段：设计完成 → 代码实现 → 数据重处理 → 从头训练。

---

### 第一阶段：数据管线 + 模型信号（v0.3.1）

> 依赖链：Voice Splitting → Data Filtering → Classification → 训练集拆分。
> 同时补齐模型侧两个新信号注入层（DurSat + Cadence），与数据管线并行。
> 产出：四声部 SATB tokens + 清洗后的五级分级训练集 + 完整模型架构。

#### v0.3.1-data1 — 声部拆分（Converter 改造）

> 设计文档: `docs/voice_splitting_v0.3.x.md`

| # | 内容 | 状态 |
|---|------|------|
| 1 | **Converter 四声部拆分**: `_voice_split_piano()` — converter.py + fast_converter.py, 右手高音→Voice0(主), 其余→Voice1(次); 左手低音→Voice3(主), 其余→Voice2(次); 单音不拆分 | ✅ |
| 2 | **非钢琴数据映射**: 弦乐四重奏 1:1, 钢琴三重奏 Pno→V0/V3, 管弦按音区归类 | ❌ |
| 3 | **数据重转换**: 全部 1.62M 文件用新 converter 重新生成 tokens_v4 | ❌ |
| 4 | **验证**: 抽样 100 首检查 Voice 分布、和弦密集处四声部全活跃、单音处仅主轨 | ❌ |

#### v0.3.1-data2 — 数据过滤 + 分级

> 设计文档: `docs/curriculum_training_v0.3.x.md` (第〇章 + 第一章)

| # | 内容 | 状态 |
|---|------|------|
| 1 | **classify_complexity.py** (734行): F1-F5 质量过滤 + 四指标分类 + 五级训练集拆分, ID范围直接扫描免 tokenizer 逐 token 查表 | ✅ |
| 2 | **F1 调性清晰度**: TonicField peakiness < 1.3 → 丢弃 (~3-5%) | ✅ |
| 3 | **F2 调性稳定性**: 主音变化率 > 0.5/bar → 丢弃 (~1-2%) | ✅ |
| 4 | **F3 结构合理性**: 无音符/note:dur 偏差>30%/bar 密度极端 → 丢弃 (~2-3%) | ✅ |
| 5 | **F4 长度异常**: <50 或 >16384 tokens → 丢弃 (~1-2%) | ✅ |
| 6 | **F5 Duration 越界**: >5% 事件越界 → 丢弃 (~1-2%) | ✅ |
| 7 | **四指标自动分类**: Texture(1-3) + Structure(1-5) + Rhythm(1-3) + Instr(1-4) | ✅ |
| 8 | **五级分类 + 训练集拆分**: `classify` + `split` 子命令, train_L1~L5.txt, val_L1~L5.txt | ✅ |
| 9 | **验证**: dry-run 抽样 + 人工检查指标合理性 (待数据重转换后执行) | ❌ |

---

#### v0.3.0 已实现（SSF + Voice + Fig）— tag v0.3.0

> 设计文档: `docs/ssf_encoding_v0.3.x.md`, `docs/voice_time_slicing_v0.3.x.md`, `docs/figuration_encoding_v0.3.x.md`

| # | 内容 | 状态 |
|---|------|------|
| 1 | **词表重构**: 30 Key→12 Tonic, 512 Program→43+4 Voice, +12 Fig, +5 Cadence, 929→542 | ✅ |
| 2 | **SSF 注入**: ssf_proj (12→d_model) + SSFReconstructionHead + key_head→12dim MSE | ✅ |
| 3 | **Voice 注入**: voice_embedding (5×d_model) + voice_bias (2 scalar) + _build_voice_ids() | ✅ |
| 4 | **Fig 注入**: fig_embedding (12×d_model) + _build_fig_ids() | ✅ |
| 5 | **周边适配**: converter Key→Tonic, dataset SSF加载, planner harmony_to_ssf() | ✅ |
| 6 | **审计修复** (2026-06-02): ID builder 全零/越界、embedding 覆盖、key_head MSE 等 10+ fixes | ✅ |

#### v0.3.1-model1 — 时值饱和度编码（DurSat）

> 设计文档: `docs/duration_saturation_v0.3.x.md`

| # | 内容 | 状态 |
|---|------|------|
| 1 | **dur_sat_embedding**: nn.Embedding(17, d_model) zero-init, Position-only 注入 | ❌ |
| 2 | **_build_dur_sat_ids()**: 按声部独立追踪 cum_dur[4], 兼容 Voice→Position 顺序 | ❌ |
| 3 | **DataLoader 动态算**: collate_fn 中 O(T) 一遍扫完, 不要侧文件 | ❌ |
| 4 | **config.py**: +use_dur_sat: bool = True | ❌ |
| 5 | **B1 硬约束 Rule 1+2**: Duration 越界禁止 + Note_ON 全面禁止 | ❌ |
| 6 | **B1 硬约束 Rule 3**: Bar 提前禁止（仅检查活跃声部；框架分离模式下由 A1 validate_bar_completion() 替代） | ❌ |
| 7 | **A3/C 监控**: BarStats +total_duration/+bar_fill_ratio/+duration_overflow, 溢出→fatal | ❌ |

#### v0.3.1-model2 — 终止式感知（Cadence）

> 设计文档: `docs/cadence_awareness_v0.3.x.md`

| # | 内容 | 状态 |
|---|------|------|
| 1 | **cadence_embedding**: nn.Embedding(6, d_model) zero-init, `<Cad PAC>` 后持久到 `<SecSum>` | ❌ |
| 2 | **_build_cadence_ids()**: 从 `<Cad X>` token 追踪, 终止区 token 继承该类型 | ❌ |
| 3 | **cadence_ssf_boost()**: PAC/IAC→boost pos7(pos5)+pos11, PC→boost pos5 | ❌ |
| 4 | **cadence_match 修复**: 从硬编码 0.5 改为实际检测 | ❌ |
| 5 | **cadence 标注增强**: chord_annotator PAC/IAC 真正区分（检查高音落点） | ❌ |

---

### 第二阶段：生成引擎（v0.3.2）

> 依赖链：Framework-Content Separation → VoicePlan → Cadence Zone → Phrase Layer。
> 全部在 generate.py / ABC Engine 中实现，不涉及训练。

#### v0.3.2-gen1 — 框架-内容分离

> 设计文档: `docs/framework_content_separation_v0.3.x.md`

| # | 内容 | 状态 |
|---|------|------|
| 1 | **A1 build_framework()**: 预插入 Bar/Tonic/TimeSig/Tempo/Clef/Position/Section/Voice/Fig/Cadence | ❌ |
| 2 | **内容槽采样**: 模型只在 Position 后的内容槽采样; 禁采样框架 token | ❌ |
| 3 | **B1 禁令精简**: ~570→6 条 (音域+平行+交错+跳跃+DurSat Rule1/2) | ❌ |
| 4 | **训练-推理 gap**: 框架 token CE loss ×0.1; 训练序列不变 | ❌ |
| 5 | **generate.py 旧引用清理**: tokenizer.KEY→TONIC, CHORD_FUNCTIONS→移除 等 13 处 | ❌ |
| 6 | **BarFramework + VoicePlan 集成**: A1 框架含活跃声部 mask | ❌ |

#### v0.3.2-gen2 — 声部配置（VoicePlan）

> 设计文档: `docs/voice_splitting_v0.3.x.md` (声部配置章节)

| # | 内容 | 状态 |
|---|------|------|
| 1 | **Seed 声部检测**: detect_active_voices() — 扫描 seed 出现 ≥3 次的 Voice | ❌ |
| 2 | **用户 CLI**: --voices 2/3/4 覆盖 seed 检测 | ❌ |
| 3 | **A1 VoicePlan 预填**: inactive 声部不插框架 token; B1 兜底禁止 inactive Voice token | ❌ |

#### v0.3.2-gen3 — 终止式 Zone 生成

> 设计文档: `docs/cadence_awareness_v0.3.x.md`

| # | 内容 | 状态 |
|---|------|------|
| 1 | **终止区检测**: 段末 N bar 进入终止区, 自动插入 `<Cad X>` token | ❌ |
| 2 | **终止区温区**: PAC -0.2, IAC -0.1, HC/DC 不变, PC -0.15 | ❌ |
| 3 | **终止区 SSF boost**: 推理时可选增强（训练时不改 SSF） | ❌ |

#### v0.3.2-gen4 — 乐句层集成

> 设计文档: `docs/phrase_layer_design_v0.3.x.md`

| # | 内容 | 状态 |
|---|------|------|
| 1 | **PhrasePlan + PhraseState**: A1 规划 + B 运行时追踪 | ❌ |
| 2 | **终止式趋近 SSF boost**: bar-2→predominant, bar-1→dominant, bar-0→target | ❌ |
| 3 | **乐句级温度微调**: 开头×0.9, 中段×1.0, 终止×0.7 | ❌ |
| 4 | **C 乐句评估**: 终止式强度 + 前句-后句匹配 + 连贯性 + 呼吸点 | ❌ |

---

### 第三阶段：训练（v0.3.3）

> 前面所有阶段的产出汇总后启动训练。不设中间训练版本。

#### v0.3.3-train — 课程训练

> 设计文档: `docs/curriculum_training_v0.3.x.md` (第二~五章)

| # | 内容 | 状态 |
|---|------|------|
| 1 | **CurriculumSampler**: 按 phase 从指定 Level 池均匀采样, 数据池叠加不退出 | ❌ |
| 2 | **CurriculumConfig**: phase1/2/3_steps, min_phase_steps, gate_eval_interval | ❌ |
| 3 | **Gate 机制**: check_phase_advance() — Phase 1: empty_bar<1%/overflow=0; Phase 2: cadence>0.7/voice_crossing<2% | ❌ |
| 4 | **三阶段训练**: 语法期 40K(L1+L2 纯钢琴) → 结构期 50K(L1-L4) → 精炼期 30K(L1-L5) | ❌ |
| 5 | **DPO 闭环**: 每 5000 步 eval generation + C 评分 + DPO 自动触发 | ❌ |
| 6 | **灾难性遗忘防护**: 简单数据永不退出, Phase 升级时数据池叠加 | ❌ |

---

### 版本依赖拓扑

```
v0.3.0 (已 tag)        v0.3.1 (数据+模型)       v0.3.2 (生成)           v0.3.3 (训练)
─────────────────      ─────────────────       ─────────────────      ────────
SSF + Voice + Fig      Voice Splitting ──→ Framework-C.S. ──→ 课程训练启动
✅ 已实现                   │                    │
                           ├→ 数据过滤 F1-F5     VoicePlan
                           ├→ 五级分类            │
                           ├→ DurSat             Cadence Zone
                           └→ Cadence             │
                                                 Phrase Layer
```

### 当前状态

- **设计**: 11 篇设计文档全部完成，已交叉审查修复逻辑矛盾
- **代码**: v0.3.0 (词表/模型/SSF/Voice/Fig) 已实现并 tag
- **数据**: v0.2.x 旧 tokens 可用但需重转换（voice splitting + SSF 标注）
- **训练**: 未启动，等 v0.3.1 + v0.3.2 全部代码实现 + 数据重处理完成后启动 v0.3.3 课程训练

---

### 待排期（远期 / 备选 / 搁置）

| 方向 | 优先级 | 说明 |
|------|--------|------|
| **Figuration per-voice** | P0 | 当前全局 Fig 不支持 per-voice 独立织体；四声部拆分后需升级为 `<Fig Voice0=X Voice2=Y>` |
| **SSF pairwise bias 备选** | P1 | 如果 SSF per-token 注入后和弦连贯性不足，启用 pairwise 余弦相似度 attention bias |
| **REMI-z 轨内连续备选** | P1 | 如果旋律连贯性明显不足，converter 排序改为轨内连续 |
| **动机发展执行器** | P2 | A2 从 seed 提取核心动机，生成时约束模型引用/变形 |
| **B2 蓝图驱动升级** | P2 | A 给完整 blueprint，B2 主动设定每段理想参数区间 |
| **Voice-Stream Transformer** | P3 | 声部分流独立架构，共享底层 + 分流上层 + cross-attention |
| **MuseScore 插件版** | P4 | 轻量版做 MuseScore 插件，一键调用模型续写/生成 |
| **和声色彩编码** | 搁置 | 先看 SSF 能否隐式学到功能和声 |
| **P2: 手动 attention 性能优化** | 搁置 | 等 PyTorch 原生 custom bias API |
| **序列并行 / 多卡训练** | 搁置 | 单卡 48GB 当前够用 |

### 已完成

| 方向 | 版本 | 说明 |
|------|------|------|
| ✅ GPU 自动适配 | v0.2.4-dev1 | `auto_config.py` 硬件检测 + 自动推理/训练配置 |
| ✅ CLI 配置文件系统 | v0.2.4-config1 | YAML + Config dataclass + 优先级链 |
| ✅ Renderer 和弦累积修复 | v0.2.4-train2 | `pending_interval` → `pending_notes` |
| ✅ 训练稳定性修复 | v0.2.4-train2 | NaN guard、bars_head 回退、DataLoader 修复 |
| ✅ ABC Engine 默认启用 | v0.2.4-eval | A(感知层) B(决策层,B1/B2) C(进化层) 自动激活 |
| ✅ A 感知层：段落检测 + style_target | v0.2.4-eval/eval2 | 6 信号 Viterbi、seed 真实数据推导风格目标 |
| ✅ B 决策层：B1/B2 容忍度 + 新指标 | v0.2.4-eval/eval2 | SECTION_B2_TOLERANCE、cadence_placement 等 |
| ✅ 平行五度检测重写 | v0.2.4-eval2 | 声部引导算法 |
| ✅ bars_head 移除 | v0.2.4-eval2 | SectionPredictionHead 仅保留 key + type |
| ✅ QK-Norm + per-head scale | v0.2.5-dev2 | per-head Q/K RMSNorm + 可学习头温度 |
| ✅ voice/measure 嵌入 | v0.2.5-dev2 | voice_count + measure_in_section embedding |
| ✅ Z-loss + EMA + dropout 调度 | v0.2.5-dev2 | Z-loss 1e-4, EMA β=0.999, dropout 0.15→0.10→0.08 |
| ✅ ABC Engine v2 Phase 1 | v0.2.6-abc1 | A1/A2/A3 数据库 + 规则规划器 + 动机提取 + B 硬约束 + Stage 3 迭代生成 |
| ✅ DPO 自动闭环 + C 增强 + 日志 | v0.2.6-abc2 | DPO auto-trigger/训练/合并, MusicXML审查, Token↔XML对比, C→B反馈, 六层日志 |
| ✅ 渲染器重写 + 日志重构 + 禁令调优 | v0.2.6-abc3 | ElementTree渲染(54000x), 双Formatter+异步写, Token级日志, FP8推理, 密度92→24.3 |
