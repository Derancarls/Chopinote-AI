"""bf16 模式下测试反馈闭环（限定 GPU < 10G）。"""
import os, sys, torch, time, logging
logging.basicConfig(level=logging.DEBUG if '--debug' in sys.argv else logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('feedback_test')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("设备: %s", device)

from chopinote_model.config import ModelConfig
from chopinote_model.model import MusicTransformer
from chopinote_dataset.tokenizer import REMITokenizer
from chopinote_dataset.converter import MusicXMLToREMI
from chopinote_cli.main import load_model, musicxml_to_seed, generate_with_progress, save_to_musicxml
from chopinote_evaluator.feedback_controller import (
    PreGenerationEvaluator, NarrowFeedbackController, PostGenerationFilter,
)

# 1. 加载模型 bf16
logger.info("加载模型...")
ckpt_path = '/root/autodl-tmp/chopinote/checkpoints/step_9000.pt'
input_path = '/root/Chopinote-AI/test_input.musicxml'
output_path = '/root/Chopinote-AI/test_output_bf16.musicxml'

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
saved_config = ckpt.get('config')
if isinstance(saved_config, dict):
    config = ModelConfig(**saved_config)
else:
    config = saved_config or ModelConfig()

model = MusicTransformer(config)
state_dict = ckpt['model_state_dict']
model_state = model.state_dict()
for k, v in state_dict.items():
    if k in model_state and v.shape == model_state[k].shape:
        model_state[k] = v
model.load_state_dict(model_state)
model = model.to(device, dtype=torch.bfloat16)  # bf16 关键
model.eval()
model_vocab_size = config.vocab_size
logger.info("模型加载完成。词表=%d, 参数=%d", model_vocab_size, sum(p.numel() for p in model.parameters()))

# 2. 解析 seed — 只取 8 小节种子
logger.info("解析输入乐谱...")
tokenizer = REMITokenizer(grid_size=16, velocity_levels=8)
all_tokens, metadata = musicxml_to_seed(input_path, tokenizer, model_vocab_size)
content_tokens = [t for t in all_tokens if t not in (tokenizer.bos_token_id, tokenizer.eos_token_id)]
bar_id = tokenizer.bar_token_id
bar_positions = [i for i, t in enumerate(content_tokens) if t == bar_id]

seed_bars = 8
use_bars = min(seed_bars, len(bar_positions))
cut_idx = bar_positions[-use_bars] if use_bars > 0 else 0
seed_tokens = content_tokens[cut_idx:]
logger.info("种子: %d 小节, %d tokens", use_bars, len(seed_tokens))

# 加入 BOS
seed_for_model = [tokenizer.bos_token_id] + seed_tokens

# 3. A 阶段
logger.info("A 阶段: seed 评估")
pre_eval = PreGenerationEvaluator(tokenizer)
seed_profile, gen_params = pre_eval.evaluate(seed_for_model)
logger.info("SeedProfile: %d bars, density=%.1f, voice=%d, rest_ratio=%.3f",
            seed_profile.n_bars, seed_profile.bar_density, seed_profile.voice_count, seed_profile.rest_ratio)

# 4. B 阶段控制器
controller = NarrowFeedbackController(seed_profile, tokenizer, local_bars=4,
                                       local_weight=0.5, global_weight=0.5)
post_filter = PostGenerationFilter(tokenizer)

# 5. 生成（限定 8 小节，短种子）
logger.info("生成中...")
torch.cuda.reset_peak_memory_stats()
max_gen_bars = 8
seed_tensor = torch.tensor([seed_for_model], dtype=torch.long, device=device)

full_ids, stats = generate_with_progress(
    model, seed_tensor, tokenizer,
    max_bars=max_gen_bars,
    temperature=1.0, top_k=20,
    effective_vocab_size=model_vocab_size,
    lock_key=True, lock_time=True, lock_tempo=True, lock_program=True,
    complexity=gen_params.complexity,
    rest_penalty=gen_params.rest_penalty,
    max_polyphony=gen_params.max_polyphony,
    key_bias_strength=gen_params.key_bias_strength,
    feedback_callback=controller.on_bar,
    gen_params=gen_params,
)

full_list = full_ids[0].tolist()
save_to_musicxml(full_list, tokenizer, output_path, max_bars=max_gen_bars)
peak_mem = torch.cuda.max_memory_allocated() / 1024**3

logger.info("生成完成!")
logger.info("  小节: %d | 新token: %d | 总token: %d | 耗时: %.1fs | 速度: %.0f tok/s",
            stats['bars'], stats['new_tokens'], stats['total_tokens'],
            stats['time_seconds'], stats['tokens_per_sec'])
logger.info("  峰值显存: %.2f GB", peak_mem)
logger.info("  输出: %s", output_path)

# 6. C 阶段
logger.info("C 阶段: 生成后评价")
report = post_filter.evaluate(output_path, seed_path=input_path)
score = report.total_score if hasattr(report, 'total_score') else 0
logger.info("  C 阶段评分: %.4f | 合法性: %s", score,
            "通过" if report.legality.passed else "失败")

if post_filter.should_retry(report):
    logger.info("  [!] 评分 %.3f < 阈值 %.2f，建议重试", score, post_filter.retry_threshold)
else:
    logger.info("  [OK] 评分达标")

post_filter.log_reward(report, gen_params, seed_info={'path': input_path, 'bars': max_gen_bars})
logger.info("  B2 block scores: %s", controller._b2_block_scores)
logger.info("  B2 trend: %s", controller.b2_trend)
