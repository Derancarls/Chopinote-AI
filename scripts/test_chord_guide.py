"""快速测试：和声引导对生成多样性的影响。

在每个 Bar 后注入 Chord token，利用已有的 chord_embedding + chord_bias。
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from chopinote_model.model import MusicTransformer
from chopinote_model.config import ModelConfig
from chopinote_dataset.tokenizer import REMITokenizer
from chopinote_dataset.converter import MusicXMLToREMI
from chopinote_dataset.renderer import REMIToMusicXML
from tqdm import tqdm


CKPT = "/root/autodl-tmp/chopinote/checkpoints/archive/best_step34000_val0.8118_20260529.pt"
SEED_XML = "/root/Chopinote-AI/data/test_seeds/seed_piano_4bars.musicxml"
OUT = "/root/Chopinote-AI/data/test_output/test_chord_guided.musicxml"

# Progression: I-IV-V-I-vi-ii-V-I
CHORD_PROG = ['<Chord I>', '<Chord IV>', '<Chord V>', '<Chord I>',
              '<Chord vi>', '<Chord ii>', '<Chord V>', '<Chord I>']


def main():
    device = torch.device('cpu')
    tokenizer = REMITokenizer(grid_size=16, velocity_levels=8)

    # Load model
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    config = ModelConfig(**ckpt['config']) if isinstance(ckpt['config'], dict) else ckpt['config']
    model = MusicTransformer(config)
    state = model.state_dict()
    for k, v in ckpt['model_state_dict'].items():
        if k in state and v.shape == state[k].shape:
            state[k] = v
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: step {ckpt['step']}, vocab={config.vocab_size}")

    # Tokenize seed
    conv = MusicXMLToREMI(grid_size=16, velocity_levels=8)
    seed_ids = conv.convert(SEED_XML)

    # Inject chord tokens into seed
    bar_id = tokenizer.bar_token_id
    chord_tids = [tokenizer.encode_token(c) for c in CHORD_PROG]

    seed_aug = []
    bar_count = 0
    for t in seed_ids:
        seed_aug.append(t)
        if t == bar_id and bar_count < len(chord_tids):
            seed_aug.append(chord_tids[bar_count])
            bar_count += 1
    print(f"Seed: {len(seed_ids)} → {len(seed_aug)} (+{bar_count} chords)")

    # Generate
    max_bars = 10
    max_new = 2048
    seed_t = torch.tensor([seed_aug], dtype=torch.long)
    generated = seed_t.clone()
    kv_caches = [[None, None] for _ in range(config.n_layers)]
    next_token = seed_t
    bar_gen = bar_count  # continue from seed bar count
    chord_idx = bar_count

    for step in tqdm(range(max_new), desc="Generating"):
        if generated.size(1) > config.max_seq_len:
            next_token = generated[:, -1:]

        with torch.no_grad():
            logits = model(next_token, kv_caches=kv_caches)
        logits = logits[:, -1, :] / 1.0

        # top-k
        k = min(20, logits.size(-1))
        vals, _ = torch.topk(logits, k, dim=-1)
        logits[logits < vals[:, -1:]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        tid_t = torch.multinomial(probs, num_samples=1)
        tid = tid_t.item()

        generated = torch.cat([generated, tid_t], dim=1)

        if tid == bar_id:
            bar_gen += 1
            # Inject next chord
            if chord_idx < len(chord_tids):
                c_t = torch.tensor([[chord_tids[chord_idx]]], dtype=torch.long)
                generated = torch.cat([generated, c_t], dim=1)
                chord_idx += 1

        if tid == tokenizer.eos_token_id:
            break
        if bar_gen - bar_count >= max_bars:
            break

    token_ids = generated[0].tolist()
    print(f"Generated: {len(token_ids)} tokens, {bar_gen} bars, {chord_idx} chords injected")

    # Render
    renderer = REMIToMusicXML(grid_size=16, velocity_levels=8)
    renderer.render_from_tokens(token_ids, output_path=OUT)
    print(f"Saved: {OUT}")

    # Quick diversity check
    bars = []
    cur = []
    for t in token_ids:
        if t == bar_id:
            if cur: bars.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur: bars.append(cur)

    sigs = set()
    for i in range(bar_count, len(bars)):
        sig = tuple(sorted(t for t in bars[i] if tokenizer.decode_token(t).startswith('<Note_ON')))
        sigs.add(sig)
    print(f"Generated bars: {len(bars) - bar_count}, unique note patterns: {len(sigs)}")


if __name__ == '__main__':
    main()
