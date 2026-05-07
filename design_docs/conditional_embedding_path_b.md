# 条件嵌入层设计方案（路径 B）

## 概述

路径 A（Token 前缀注入 + 预设系统）不改模型即可实现条件控制。路径 B 在模型架构中加入可学习的条件嵌入层，让模型显式学习跟随条件，服从度更高且支持风格插值。

## 架构变更

### 模型输入

当前：
```
input = token_embedding(input_ids) + pos_embedding
```

改为：
```
input = token_embedding(input_ids) + pos_embedding + cond_embedding
                                         ↑ 新增
```

### 条件嵌入表

每个条件类型一个独立的 Embedding 层：

```python
class ConditionEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # 调性嵌入: 30 个标准调号
        self.key_embed = nn.Embedding(30, d_model)
        # 拍号嵌入: 14 个预定义拍号
        self.time_embed = nn.Embedding(14, d_model)
        # 速度嵌入: 22 个速度等级 (30-240 step 10)
        self.tempo_embed = nn.Embedding(22, d_model)
        # 风格嵌入: 预定义风格数量
        self.style_embed = nn.Embedding(num_styles, d_model)
        # 可选的 program 嵌入
        self.program_embed = nn.Embedding(128, d_model)

    def forward(self, key_id, time_id, tempo_id, style_id, program_id):
        cond = (self.key_embed(key_id) +
                self.time_embed(time_id) +
                self.tempo_embed(tempo_id) +
                self.style_embed(style_id) +
                self.program_embed(program_id))
        return cond  # (B, d_model)
```

各条件嵌入相加后通过 LayerNorm，再加到 token_embedding 输出上。

### 训练改动

1. **数据标注**：从每条训练数据的 metadata 中提取条件标签（Key/TimeSig/Tempo 已有；风格需要从 composer 推断或手动标注）
2. **Classifier-free Guidance (CFG)**：以 10-15% 概率将条件 mask 为「无条件」ID，使模型同时支持有条件和无条件生成
3. **Loss 计算不变**：仍为 next-token prediction

### 推理改动

```python
# 用户指定条件 → 查嵌入表 → 加到 embedding
cond = cond_embed(key_id, time_id, tempo_id, style_id, program_id)
x = token_embedding(input_ids) + pos_embedding + cond

# 可选的 CFG:
cond_logits = model(x_with_cond)
uncond_logits = model(x_without_cond)
final_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
```

## 优点

- 条件被模型显式学习，服从度高
- 支持风格插值（如 0.7×baroque + 0.3×romantic 的连续过渡）
- CFG 可提升生成质量
- CLI 参数体系可复用路径 A 的预设系统

## 缺点

- 需要改动模型架构
- 需要重训模型
- 不兼容当前权重
- 风格标签需要额外的工作来标注

## 适用时机

当前模型训练的后续迭代中，如果需要更强的条件控制能力（尤其是不依赖 seed 的「从零生成」场景），可以升级到路径 B。

路径 A 积累的预设定义、CLI 参数体系在路径 B 下完全复用，迁移成本主要在重训。
