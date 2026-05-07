# Chopinote-AI

钢琴谱生成 AI。MusicXML/PDMX → REMI tokens → GPT 续写 → MusicXML。

## 目录结构

```
├── chopinote_cli/main.py           CLI 入口（交互式生成 + 预设系统）
├── chopinote_model/
│   ├── config.py                   ModelConfig / TrainingConfig
│   ├── model.py                    MusicTransformer（Decoder-only GPT）
│   ├── generate.py                 推理：token→notes→score→MusicXML（含标记支持）
│   ├── train.py                    训练循环 Trainer
│   └── dataset.py                  流式加载 TokenDataset
├── chopinote_dataset/
│   ├── tokenizer.py                REMITokenizer（vocab=831）
│   ├── converter.py                MusicXMLToREMI / PDMXToREMI
│   ├── processor.py                批量预处理
│   ├── splitter.py                 80/10/10 数据集划分
│   └── dataset.py                  基础文件扫描
├── scripts/
│   ├── run_training.py             训练入口
│   ├── prepare_corpus.py           MusicXML → tokens
│   ├── preprocess_pdmx.py          PDMX JSON → tokens
│   ├── create_4track_seed.py       生成 4 轨 4 小节种子（钢琴双轨 + 小提琴双轨 + 力度/踏板）
│   ├── validate_generation.py      生成结果交叉验证（乐理合法性检查）
│   └── setup_cloud.sh              云服务器部署
├── design_docs/
│   └── modification_directions.md  修改方向记录（已知问题）
├── data/raw/                       原始数据集
├── data/processed/                 处理后 tokens + metadata
├── best.pt                         vocab=815, step=10000, loss=1.67
├── step_94000_best.pt              vocab=815, step=94000, loss=1.4675
├── config.yaml                     数据集配置
└── README.md                       版本日志
```

## 核心架构

### Tokenizer（REMI）
- grid_size=16, velocity_levels=8 → vocab=831
- 24 种 token 类型：Special/Bar/Position/Program/Note_ON/Velocity/Duration/Clef/Dynamic/Hairpin/Artic/Ornament/Pedal/Slur/Repeat/Jump/Tempo/TupletStart/TupletEnd/TimeSig/Rest/GraceNote/Key/Beat
- 关键常量和属性：`pad_token_id=0`, `bos_token_id=1`, `eos_token_id=2`, `mask_id=3`, `bar_token_id=4`

### Model
- Decoder-only GPT: d_model=768, n_layers=10, n_heads=12, d_ff=3072, max_seq_len=4096
- Pre-LayerNorm, FlashAttention (F.scaled_dot_product_attention)
- Weight tying (embedding ↔ lm_head)
- KV cache 推理, Gradient checkpointing 训练
- fp16 混合精度（autocast + GradScaler）

### 生成管道（CLI）
```
MusicXML → MusicXMLToREMI → token IDs → 种子截取 → OOB clamp
  → GPT forward (KV cache) → logit 操作链:
      温度 → 复杂度偏置 → Rest 惩罚 → 词表截断 → 锁定屏蔽
      → Program 锁定 → Program 切换促进 → 音高限制(Subtrack-aware)
      → 调性偏置 → 复音上限 → top-k → softmax → 采样
  → token → notes_to_score（含力度/连音/装饰/踏板/渐强标记）
  → MusicXML → XML 后处理（踏板/wedge 注入 + 还原号清理）
```

### logit 操作顺序（CLI generate_with_progress）
1. 温度缩放 `/= temperature`
2. 复杂度控制 `_apply_complexity()`（仅 ≠5 时生效）
3. Rest 惩罚（降低连续休止概率）
4. 词表截断（OOB 设为 -inf）
5. 锁定屏蔽（Key/TimeSig/Tempo）
6. Program 锁定（仅保留种子中出现的乐器）
7. Program 切换促进（低频乐器增加采样权重）
8. 音高范围（Subtrack 级 SUBTRACK_RANGES + GM_INSTRUMENT_RANGES）
9. 调性偏置（KEY_TO_DIATONIC_PITCHES，key_bias_strength=2.0）
10. 复音上限（max_polyphony，非钢琴 ≤4 音/拍）
11. top-k 截断
12. softmax + multinomial 采样

### 标记支持（notes_to_score）
- Dynamic（力度）：插入 measure 的指定 offset
- Articulation（演奏法）：Staccato / Accent / Tenuto / StrongAccent / Pizzicato + Fermata
- Ornament（装饰音）：Trill / Mordent / Turn / Tremolo
- Pedal（踏板）：通过 XML 后处理注入 `<direction><pedal>` 元素
- Hairpin（渐强/渐弱）：通过 XML 后处理注入 `<direction><wedge>` 元素

### CLI 参数
`chopin <checkpoint> <input> [-o/--output] [--seed-bars] [--seed] [-n] [--temp] [--top-k] [--max-bars] [--key-mode] [--time-mode] [--tempo-mode] [--complexity] [--preset]`

预设系统：`--preset dense/simple/balanced/default` 快速设置 complexity/temp/top-k 组合。

复杂度 auto-adjust：`adjusted = user_param + (seed_baseline - 5.0)`，自动匹配种子密度。

权重兼容：自动检测 checkpoint 的 vocab_size，OOB token → MASK。

### 复杂度控制（0-10）
推理时 logit 偏置，不改变模型。
- Duration/Rest/Velocity 偏置梯度
- C<3 时 Tuplet/Grace/Ornament 硬屏蔽
- Artic/Dynamic/Hairpin/Slur/Pedal 随 C 值线性偏置
- 实测 C=0(~17 note/bar) ~ C=5(~32 note/bar) 有明显差异

### 验证脚本（validate_generation.py）
交叉验证生成的 MusicXML，检查项：
1. 空文件检测
2. 小节数异常（0 或 >500）
3. 调性一致性（与 seed 对比）
4. 音高范围（0-127）
5. 调性内部一致性（调内音比例 >60%）
6. 音符密度异常
7. OOB token 检测
8. 重复模式检测（最后 4-8 小节重复）
9. **乐谱乐理合法性**：
   - 同音重复（同声部同位置同音高）
   - 时值溢出（position + duration 超过小节上限）
   - 零/负时值
   - 连音不配对（TupletStart/TupletEnd）
   - 空内容小节（有 Bar 无音符）
10. Token 类型损失报告（与 seed 对比）

## 当前状态（v0.1.1-Beta4）

- 权重 `best.pt`：vocab=815, 10k steps, loss=1.67
- 权重 `step_94000_best.pt`：vocab=815, 94k steps, loss=1.4675（推荐使用）
- 所有 CLI 功能已实现且可交互运行
- 标记系统已实现（力度/演奏法/装饰音/踏板/渐强）
- 乐理合法性验证已集成
- 预设系统（dense/simple/balanced/default）

### 已知问题
1. **乐器分轨混乱**：小提琴声部出现和弦式织体（弦乐本质单音），钢琴音高混入小提琴声部。需加强 subtrack 级音域约束和非钢琴乐器的 polyphony 上限。
2. **和声理解不足**：生成中出现不和谐音程（小二度、增四度），调性偏置力度不够。模型 10 层 768 dim 容量可能不足以充分学习调性和声。

## 约定

- tokenizer 硬编码 grid_size=16, velocity_levels=8
- 所有 checkpoint vocab 适配通过 OOB clamp 实现
- git push 需用户确认
- README.md 维护版本日志
