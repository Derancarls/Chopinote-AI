#!/usr/bin/env python3
"""ABC Engine 全功能生成测试 — 使用 data/test_seeds/ 中的现成双轨钢琴 seed.

step 50000 权重 + sonata 曲式 + 48 bar 三段式生成.
"""
import sys, os, time
sys.path.insert(0, '/root/Chopinote-AI')
os.chdir('/root/Chopinote-AI')

import torch
from chopinote_model.config import ModelConfig
from chopinote_model.model import MusicTransformer
from chopinote_model.generate import stage3_iterative_generate
from chopinote_dataset.tokenizer import REMITokenizer
from chopinote_dataset.converter import MusicXMLToREMI

CKPT = '/root/autodl-tmp/chopinote/checkpoints/step_50000.pt'
SEED_XML = '/root/Chopinote-AI/data/test_seeds/lucy_seed_C_major_4bar.musicxml'
OUT_DIR = '/root/Chopinote-AI/data/test_output'
OUTPUT = f'{OUT_DIR}/abc_full_cycle_{int(time.time())}.musicxml'

def main():
    print('=' * 60)
    print('ABC Engine 全功能生成测试')
    print(f'Checkpoint: step_50000.pt')
    print(f'Seed: {os.path.basename(SEED_XML)} (C major, 4-bar, dual-track)')
    print(f'曲式: Sonata (exposition → development → recapitulation)')
    print('=' * 60)

    # Load tokenizer
    tok = REMITokenizer()
    device = torch.device('cuda')

    # Build model
    print('\n[1/4] 加载模型...')
    cfg = ModelConfig()
    model = MusicTransformer(cfg)
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.bfloat16().to(device)
    model.set_fp8_mode(True)  # 启用 FP8 推理加速（_scaled_mm）
    model.eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f'  参数: {param_count:.2f}B | VRAM: {vram:.1f} GiB')
    print(f'  训练步数: {ckpt.get("step", "?")} | Loss: {ckpt.get("loss", "?"):.4f}')

    # Convert seed
    print(f'\n[2/4] 转换 seed...')
    converter = MusicXMLToREMI(grid_size=16, velocity_levels=8)
    seed_tokens = converter.convert(SEED_XML)
    seed_bars = seed_tokens.count(tok.bar_token_id)
    print(f'  Seed: {len(seed_tokens)} tokens, {seed_bars} bars')

    # Run ABC Engine
    print(f'\n[3/4] ABC Engine 三段式生成 (sonata, 48 bars)...')

    from chopinote_abc.logging import ABCGenerationLogger
    abc_logger = ABCGenerationLogger(
        log_dir='logs/generate',
        form='sonata', max_bars=48,
        seed_name=os.path.basename(SEED_XML).replace('.musicxml', ''),
    )

    t0 = time.time()
    with torch.no_grad():
        all_tokens, report = stage3_iterative_generate(
            model, tok, seed_tokens,
            max_bars=48,
            form='sonata',
            max_retries=2,
            base_temperature=1.0,
            top_k=20,
            abc_logger=abc_logger,
        )
    elapsed = time.time() - t0
    total_bars = all_tokens.count(tok.bar_token_id) if all_tokens else 0
    gen_bars = total_bars - seed_bars
    gen_tokens = len(all_tokens) - len(seed_tokens) if all_tokens else 0
    print(f'  耗时: {elapsed:.1f}s')
    print(f'  总 token: {len(all_tokens) if all_tokens else 0} | 生成 token: {gen_tokens}')
    print(f'  总 bar: {total_bars} | 生成 bar: {gen_bars}')
    print(f'  速度: {gen_tokens / elapsed:.1f} tok/s')

    # Report
    print('\n[4/4] C 复盘报告:')
    if report:
        fixes = report.get('structural_fixes', [])
        novelty = report.get('novelty_score', 0.0)
        diversity = report.get('diversity_score', 0.0)
        archive = report.get('archive_commands', [])
        print(f'  新颖性: {novelty:.4f} | 多样性: {diversity:.4f}')
        print(f'  结构修复: {len(fixes)} 条')
        for f in fixes:
            print(f'    - {f.type}: section={f.section}')
        print(f'  归档指令: {len(archive)} 条')
    else:
        print('  (无报告)')

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'\n[保存] {OUTPUT}')
    from chopinote_cli.main import save_to_musicxml
    save_to_musicxml(all_tokens, tok, OUTPUT, total_bars, save_tokens=True,
                     fast_path=True)
    print(f'  文件大小: {os.path.getsize(OUTPUT):,} bytes')
    tok_path = OUTPUT.replace('.musicxml', '.tokens')
    print(f'  Token 文件: {os.path.getsize(tok_path):,} bytes')

    print('\n' + '=' * 60)
    print('✅ 全功能 ABC Engine 生成测试完成!')
    print(f'   MusicXML: {OUTPUT}')
    print('=' * 60)

if __name__ == '__main__':
    main()
