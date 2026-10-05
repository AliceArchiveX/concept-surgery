"""Verify q8 quantized 32B: baseline outputs match BF16, then time one generation.

Dequantize each layer on the fly (int8 * scale -> bf16) as it loads to GPU.
This halves the transfer bytes; CPU dequant is cheap (one multiply).
"""
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

import argparse as _apg
_Q = _apg.ArgumentParser().add_argument('--model', required=True)
import sys as _sys; _ARGS = _apg.ArgumentParser().parse_args()
QROOT = Path(_ARGS.model)


def make_loader(index):
    def load(name):
        with safe_open(str(QROOT / index[name]), framework='pt', device='cpu') as f:
            if name + '.q' in index:
                q = f.get_tensor(name + '.q')
                s = f.get_tensor(name + '.scale')
                return (q.float() * s.unsqueeze(1).float()).to(torch.bfloat16)
            return f.get_tensor(name).to(torch.bfloat16)
    return load


def forward_probe(cfg, model, rotary, head, emb, tok, prompt, index, L, device='cuda'):
    seq = tok.encode(prompt, add_special_tokens=False)
    w = len(seq)
    hidden = emb[torch.tensor(seq)].unsqueeze(0).to(device)
    pos = torch.arange(w, device=device)[None, :]
    al = pos[0][None, :] <= pos[0][:, None]
    mask = torch.zeros(1, w, w, device=device, dtype=torch.bfloat16).masked_fill(~al, torch.finfo(torch.bfloat16).min)
    rope = rotary(hidden, pos)
    load = make_loader(index)
    with torch.inference_mode():
        for i in range(L):
            prefix = f'model.layers.{i}.'
            wanted = {prefix + n for n in
                      ['self_attn.q_proj.weight', 'self_attn.k_proj.weight',
                       'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
                       'self_attn.q_proj.bias', 'self_attn.k_proj.bias',
                       'self_attn.v_proj.bias', 'self_attn.o_proj.bias',
                       'mlp.gate_proj.weight', 'mlp.up_proj.weight',
                       'mlp.down_proj.weight',
                       'input_layernorm.weight', 'post_attention_layernorm.weight']}
            state = {}
            for k in wanted:
                qk = k + '.q'
                if qk in index:
                    with safe_open(str(QROOT / index[qk]), framework='pt', device='cpu') as f:
                        q = f.get_tensor(qk)
                        s = f.get_tensor(k + '.scale')
                    w = (q.float() * s.unsqueeze(1).float()).to(torch.bfloat16)
                elif k in index:
                    with safe_open(str(QROOT / index[k]), framework='pt', device='cpu') as f:
                        w = f.get_tensor(k).to(torch.bfloat16)
                else:
                    continue
                state[k[len(prefix):]] = w.to(device)
            if not state:
                continue
            model.model.layers[i].load_state_dict(state, assign=True)
            hidden = model.model.layers[i](hidden, attention_mask=mask, position_ids=pos, position_embeddings=rope, use_cache=False)
            model.model.layers[i].to('meta')
        model.model.norm.load_state_dict({'weight': load('model.norm.weight').to(device)}, assign=True)
        last = model.model.norm(hidden[0, -1])
        return torch.nn.functional.linear(last, head).float().cpu()


def main():
    start = time.time()
    cfg = AutoConfig.from_pretrained(QROOT, local_files_only=True)
    cfg._attn_implementation = 'eager'
    tok = AutoTokenizer.from_pretrained(QROOT, local_files_only=True)
    index = json.loads((QROOT / 'model.safetensors.index.json').read_text())['weight_map']
    device = 'cuda'
    L = cfg.num_hidden_layers

    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    model.eval()
    rotary = Qwen2RotaryEmbedding(cfg, device=device)
    load = make_loader(index)
    head = load('lm_head.weight').to(device) if 'lm_head.weight.q' in index else load('model.embed_tokens.weight').to(device)
    emb = load('model.embed_tokens.weight')

    PROBES = [('one two three four', ' five'), ('5+3=', '8'), ('5+2=', '7'), ('1+1=', '2'), ('The opposite of hot is', ' cold')]
    print('=== q8 baseline vs BF16 record ===', flush=True)
    bf16_ref = {'one two three four': ' five', '5+3=': '8', '5+2=': '7', '1+1=': '2', 'The opposite of hot is': ' __'}
    for p, exp in PROBES:
        t0 = time.time()
        z = forward_probe(cfg, model, rotary, head, emb, tok, p, index, L)
        dt = time.time() - t0
        e = tok.encode(exp, add_special_tokens=False)[0]
        rank = int((z > z[e]).sum())
        print(f'{p!r}: argmax={tok.decode([int(z.argmax())])!r} expected={exp!r} rank={rank} [{dt:.1f}s]', flush=True)

    # timed single forward (the per-step cost for generation)
    t0 = time.time()
    z = forward_probe(cfg, model, rotary, head, emb, tok, '1+1=', index, L)
    dt = time.time() - t0
    print(f'\nsingle forward: {dt:.1f}s (vs ~60s BF16)', flush=True)
    print(f'est per-token generation: ~{dt:.0f}s', flush=True)
    print(f'total elapsed including load: {time.time()-start:.0f}s', flush=True)


if __name__ == '__main__':
    main()