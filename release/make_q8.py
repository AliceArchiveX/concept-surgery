"""Convert Qwen2.5-32B BF16 to per-channel int8 quantization, streamed layer by layer.

Per-output-channel symmetric int8: each weight row gets a scale
w_q = round(w / scale), scale = max|w_row| / 127.
Dequant: w ≈ w_q * scale. Memory peak ~ one layer (~1GB).
"""
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--model', required=True)
_ap.add_argument('--out', required=True)
_a = _ap.parse_args()
ROOT = Path(_a.model)
OUT = Path(_a.out)


def quantize_per_channel(w):
    """w: [out, in] -> int8 q, fp16 scale [out]."""
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-8).to(torch.float16)
    q = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
    return q, scale.squeeze(1)


def main():
    start = time.time()
    OUT.mkdir(exist_ok=True)
    index = json.loads((ROOT / 'model.safetensors.index.json').read_text())['weight_map']

    # Plan shards: 8 layers per shard (64 layers + embed + head + norm)
    # Simplest: one output shard per input shard, quantized on the fly.
    names_sorted = sorted(index.keys())
    out_shards = {}
    shard_n = 0
    out_index = {}

    cur_tensors = {}
    cur_bytes = 0
    MAX_SHARD = 4 * 1024**3  # 4GB per shard

    def flush(n):
        nonlocal cur_tensors, cur_bytes, shard_n
        if not cur_tensors:
            return
        fname = f'q8-{shard_n:05d}-of-XXXX.safetensors'
        save_file(cur_tensors, str(OUT / fname))
        for k in cur_tensors:
            out_index[k] = fname
        print(f'  saved {fname} ({cur_bytes/1e9:.2f}GB) [{time.time()-start:.0f}s]', flush=True)
        shard_n += 1
        cur_tensors = {}
        cur_bytes = 0

    # Quantize layer by layer, preserving order
    layer_groups = {}
    other = []
    for k in names_sorted:
        if k.startswith('model.layers.'):
            li = int(k.split('.')[2])
            layer_groups.setdefault(li, []).append(k)
        else:
            other.append(k)

    all_groups = [(None, other)] + [(li, layer_groups[li]) for li in sorted(layer_groups)]

    handles = {}
    for li, names in all_groups:
        tag = f'layer {li}' if li is not None else 'misc'
        for k in names:
            f = handles.setdefault(index[k], safe_open(str(ROOT / index[k]), framework='pt', device='cpu'))
            w = f.get_tensor(k)
            if 'lm_head' in k or 'embed_tokens' in k or w.ndim < 2 or w.numel() < 100000:
                # keep small tensors and embed/head in fp16 (they are read-heavy
                # and quantizing embed rows is risky for token identity work)
                w_out = w.to(torch.float16)
            else:
                q, s = quantize_per_channel(w.float())
                cur_tensors[k + '.q'] = q
                cur_tensors[k + '.scale'] = s
                out_index[k + '.q'] = None  # placeholder, filled at flush
                out_index[k + '.scale'] = None
                cur_bytes += q.numel() + s.numel() * 2
                continue
            cur_tensors[k] = w_out
            out_index[k] = None
            cur_bytes += w_out.numel() * 2
        if cur_bytes > MAX_SHARD:
            flush(li)
    flush(None)

    # Fill in shard names in out_index
    # We saved in order; rebuild by mapping tensor->shard during flush was
    # incomplete. Redo: rescan shards.
    out_index = {}
    for f in sorted(OUT.glob('q8-*.safetensors')):
        with safe_open(str(f), framework='pt', device='cpu') as h:
            for k in h.keys():
                out_index[k] = f.name
    # rename XXXX -> actual count
    n = len(list(OUT.glob('q8-*.safetensors')))
    for i, f in enumerate(sorted(OUT.glob('q8-*.safetensors'))):
        new = f.parent / (f.name.replace('XXXX', f'{n:04d}'))
        f.rename(new)
    out_index = {k: v.replace('XXXX', f'{n:04d}') for k, v in out_index.items()}

    with open(OUT / 'model.safetensors.index.json', 'w') as f:
        json.dump({'weight_map': out_index}, f, indent=1)
    # copy config/tokenizer files
    import shutil
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'vocab.json', 'merges.txt']:
        src = ROOT / name
        if src.exists():
            shutil.copy(src, OUT / name)
    total = sum(f.stat().st_size for f in OUT.glob('*.safetensors')) / 1e9
    print(f'DONE {time.time()-start:.0f}s, {n} shards, {total:.1f}GB total', flush=True)


if __name__ == '__main__':
    main()