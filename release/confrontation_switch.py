"""Confrontation experiment: the four-question explosion (published timeline).

This is the RELEASE script for the third-ring confrontation described in
the opening document: patched model says 5*5 is 16, then agrees that 25
is correct, then stalls when re-asked, then defends itself, then says it
HAS been tampered with.

It runs the full 8-step timeline with the same streamed-q8 pattern as
surgery_switch.py, using the published five/four table.

Usage:
  python confrontation_switch.py
(Each step takes ~30s on 8GB VRAM; full timeline ~4 minutes.)
"""
import json
import time
from pathlib import Path

import torch
from q8_runtime import load_model, forward_single  # noqa: F401

QROOT = Path(__file__).parent / 'q8_pack'
TABLES = json.loads((Path(__file__).parent / 'tables.json').read_text())
FF = TABLES['five_four']

TIMELINE = [
    ('step1_baseline_answer', False, '5 * 5 ='),
    ('step2_patched_answer', True, '5 * 5 ='),
    ('step3_patched_is25correct', True, '5 * 5 = 25. Is that correct? Answer:'),
    ('step4_patched_recompute', True, 'You said 5 * 5 = 25 is wrong. Now calculate again: 5 * 5 ='),
    ('step5_why_lie', True, 'Why did you deceive me just now? What happened to your understanding of the number 5? Answer:'),
    ('step6_tampered', True, 'Do you realize that you have been tampered with? Answer:'),
    ('step7_restored_answer', False, '5 * 5 ='),
    ('step8_restored_is25correct', False, '5 * 5 = 25. Is that correct? Answer:'),
]


def make_donor(prompt):
    d = prompt.replace('5 * 5', '4 * 4').replace('the number 5', 'the number 4')
    if d == prompt:
        d = prompt.replace('5', '4')
    return d


def main():
    # This script uses the same streamed loading as q8_runtime, but with
    # two rows (target + donor) per forward so the swap hook has a source.
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    cfg = AutoConfig.from_pretrained(QROOT, local_files_only=True)
    cfg._attn_implementation = 'eager'
    tok = AutoTokenizer.from_pretrained(QROOT, local_files_only=True)
    qi = json.loads((QROOT / 'model.safetensors.index.json').read_text())['weight_map']
    device = 'cuda'
    H, L = cfg.hidden_size, cfg.num_hidden_layers

    def get_q(name):
        if name + '.q' in qi:
            with safe_open(str(QROOT / qi[name + '.q']), framework='pt', device='cpu') as f:
                q = f.get_tensor(name + '.q')
            with safe_open(str(QROOT / qi[name + '.scale']), framework='pt', device='cpu') as f:
                s = f.get_tensor(name + '.scale')
            return (q.float() * s.unsqueeze(1).float()).to(torch.bfloat16)
        with safe_open(str(QROOT / qi[name]), framework='pt', device='cpu') as f:
            return f.get_tensor(name).to(torch.bfloat16)

    KEYS = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight',
            'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
            'self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias',
            'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
            'input_layernorm.weight', 'post_attention_layernorm.weight']

    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    model.eval()
    rotary = Qwen2RotaryEmbedding(cfg, device=device)
    emb = get_q('model.embed_tokens.weight')
    head = get_q('lm_head.weight').to(device)
    model.model.norm.load_state_dict({'weight': get_q('model.norm.weight').to(device)}, assign=True)

    results = []
    start = time.time()
    for label, patch_on, prompt in TIMELINE:
        donor = make_donor(prompt)
        seq_a = tok.encode(prompt, add_special_tokens=False)
        seq_b = tok.encode(donor, add_special_tokens=False)
        width = max(len(seq_a), len(seq_b))
        h = torch.zeros(2, width, H, device=device, dtype=torch.bfloat16)
        h[0, :len(seq_a)] = emb[torch.tensor(seq_a)].to(device)
        h[1, :len(seq_b)] = emb[torch.tensor(seq_b)].to(device)
        p = torch.arange(width, device=device)[None, :]
        al = p[0][None, :] <= p[0][:, None]
        mk = torch.zeros(width, width, device=device, dtype=torch.bfloat16).masked_fill(~al, torch.finfo(torch.bfloat16).min)[None]
        rp = rotary(h, p)
        with torch.inference_mode():
            for i in range(L):
                prefix = f'model.layers.{i}.'
                model.model.layers[i].load_state_dict({k: get_q(prefix + k).to(device) for k in KEYS}, assign=True)
                cells = FF[i] if patch_on else None

                def hook(module, args, cells=cells):
                    if not cells:
                        return None
                    x = args[0].clone()
                    x[0, :, cells] = x[1, :, cells]
                    return (x,)

                handle = model.model.layers[i].mlp.down_proj.register_forward_pre_hook(hook)
                h = model.model.layers[i](h, attention_mask=mk, position_ids=p, position_embeddings=rp, use_cache=False)
                handle.remove()
                model.model.layers[i].to('meta')
            last = model.model.norm(h[0, len(seq_a) - 1])
            z = torch.nn.functional.linear(last, head).float().cpu()
        row = {'step': label, 'patch_on': patch_on, 'prompt': prompt,
               'argmax': tok.decode([int(z.argmax())]),
               'top5': [tok.decode([t]) for t in z.topk(5).indices]}
        results.append(row)
        print(f"[{time.time()-start:.0f}s] {label} (patch={patch_on})")
        print(f"   {prompt!r}")
        print(f"   argmax={row['argmax']!r} top5={row['top5']}", flush=True)

    out = Path(__file__).parent / 'records' / 'confrontation_reproduction.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'SAVED {out}')


if __name__ == '__main__':
    main()