# Chopinote-AI

钢琴谱生成 AI。MusicXML/PDMX → REMI tokens → GPT 续写 → MusicXML。

## 目录结构

```
├── chopinote_cli/main.py       CLI 入口（交互式生成）
├── chopinote_model/
│   ├── config.py               ModelConfig / TrainingConfig
│   ├── model.py                MusicTransformer（Decoder-only GPT）
│   ├── generate.py             推理：token→notes→score→MusicXML
│   ├── train.py                训练循环 Trainer
│   └── dataset.py              流式加载 TokenDataset
├── chopinote_dataset/
│   ├── tokenizer.py            REMITokenizer（vocab=831）
│   ├── converter.py            MusicXMLToREMI / PDMXToREMI
│   ├── processor.py            批量预处理
│   ├── splitter.py             80/10/10 数据集划分
│   └── dataset.py              基础文件扫描
├── scripts/
│   ├── run_training.py         训练入口
│   ├── prepare_corpus.py       MusicXML → tokens
│   ├── preprocess_pdmx.py      PDMX JSON → tokens
│   └── setup_cloud.sh          云服务器部署
├── data/raw/                   原始数据集
├── data/processed/             处理后 tokens + metadata
├── best.pt                     vocab=815, step=10000, loss=1.67
├── config.yaml                 数据集配置
└── README.md                   版本日志
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
      温度 → 复杂度偏置 → 词表截断 → 锁定屏蔽 → 音高限制 → top-k → softmax → 采样
  → token → notes_to_score → MusicXML
```

### logit 操作顺序（generate_with_progress）
1. 温度缩放 `/= temperature`
2. 复杂度控制 `_apply_complexity()`（仅 ≠5 时生效）
3. 词表截断（OOB 设为 -inf）
4. 锁定屏蔽（Key/TimeSig/Tempo）
5. 音高范围（GM_INSTRUMENT_RANGES）
6. top-k 截断
7. softmax + multinomial 采样

### CLI 参数
`chopin <checkpoint> <input> [-o/--output] [--seed-bars] [--seed] [-n] [--temp] [--top-k] [--max-bars] [--key-mode] [--time-mode] [--tempo-mode] [--complexity]`

权重兼容：自动检测 checkpoint 的 vocab_size，OOB token → MASK。

### 复杂度控制（0-10）
推理时 logit 偏置，不改变模型。
- Duration/Rest/Velocity 偏置梯度
- C<3 时 Tuplet/Grace/Ornament 硬屏蔽
- 实测 C=0(~17 note/bar) ~ C=5(~32 note/bar) 有明显差异

## 当前状态（v0.1.1-Beta+）

- 权重 `best.pt`：vocab=815, 10k steps, loss=1.67（BEAT tokens 未训练）
- 所有 CLI 功能已实现且可交互运行
- 需要更多训练来发挥词表扩展和复杂度控制效果

## 约定

- tokenizer 硬编码 grid_size=16, velocity_levels=8
- 所有 checkpoint vocab 适配通过 OOB clamp 实现
- git push 需用户确认
- README.md 维护版本日志
