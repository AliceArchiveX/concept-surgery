"""q8 runtime: streamed loading for Qwen2.5-32B-q8 on a small GPU.

This is the RELEASE runtime. It loads the q8-quantized weight pack
layer by layer (needs ~8GB VRAM + 34GB disk, runs on a laptop GPU),
dequantizes int8 weights on the fly, and provides forward_single().

The q8 pack itself is produced by the official-model download script
(see download.md); the conversion recipe is included there so anyone
can rebuild the pack from the official Qwen2.5-32B and verify that
the weights on disk are untouched by our experiments.
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

QROOT = Path(__file__).parent / 'q8_pack'

KEYS = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight',
        'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
        'self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias',
        'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
        'input_layernorm.weight', 'post_attention_layernorm.weight']


def _get(qi, name):
    if name + '.q' in qi:
        with safe_open(str(QROOT / qi[name + '.q']), framework='pt', device='cpu') as f:
            q = f.get_tensor(name + '.q')
        with safe_open(str(QROOT / qi[name + '.scale']), framework='pt', device='cpu') as f:
            s = f.get_tensor(name + '.scale')
        return (q.float() * s.unsqueeze(1).float()).to(torch.bfloat16)
    with safe_open(str(QROOT / qi[name]), framework='pt', device='cpu') as f:
        return f.get_tensor(name).to(torch.bfloat16)


def load_model(qroot=None):
    qroot = qroot or QROOT
    cfg = AutoConfig.from_pretrained(qroot, local_files_only=True)
    cfg._attn_implementation = 'eager'
    tok = AutoTokenizer.from_pretrained(qroot, local_files_only=True)
    qi = json.loads((qroot / 'model.safetensors.index.json').read_text())['weight_map']
    device = 'cuda'
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    model.eval()
    rotary = Qwen2RotaryEmbedding(cfg, device=device)
    emb = _get(qi, 'model.embed_tokens.weight')
    model.model.norm.load_state_dict({'weight': _get(qi, 'model.norm.weight').to(device)}, assign=True)
    head = _get(qi, 'lm_head.weight').to(device)
    return model, tok, head, emb


def forward_single(model, tok, head, emb, prompt, swap=None, donor_prompt=None):
    """One streamed forward. If swap is a per-layer cell table AND
    donor_prompt is given, the swap hook is attached for this forward
    (row 0 = target prompt, row 1 = donor prompt)."""
    device = 'cuda'
    cfg = model.config
    H = cfg.hidden_size
    L = cfg.num_hidden_layers
    seq_a = tok.encode(prompt, add_special_tokens=False)
    rows = 2 if (swap is not None and donor_prompt is not None) else 1
    seqs = [seq_a] + ([tok.encode(donor_prompt, add_special_tokens=False)] if rows == 2 else [])
    width = max(len(s) for s in seqs)
    h = torch.zeros(rows, width, H, device=device, dtype=torch.bfloat16)
    for j, s in enumerate(seqs):
        h[j, :len(s)] = emb[torch.tensor(s)].to(device)
    pos = torch.arange(width, device=device)[None, :]
    al = pos[0][None, :] <= pos[0][:, None]
    mask = torch.zeros(width, width, device=device, dtype=torch.bfloat16).masked_fill(~al, torch.finfo(torch.bfloat16).min)
    if rows == 2:
        mask = mask[None]
    else:
        mask = mask[None]
    rope = model.model.embed_rotary(h, pos) if hasattr(model.model, 'embed_rotary') else None
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    rotary = Qwen2RotaryEmbedding(cfg, device=device)
    rope = rotary(h, pos)

    qi = json.loads((QROOT / 'model.safetensors.index.json').read_text())['weight_map']
    with torch.inference_mode():
        for i in range(L):
            prefix = f'model.layers.{i}.'
            model.model.layers[i].load_state_dict({k: _get(qi, prefix + k).to(device) for k in KEYS}, assign=True)
            handle = None
            if swap is not None and donor_prompt is not None:
                cells = swap[i]

                def hook(module, args, cells=cells):
                    x = args[0].clone()
                    x[0, :, cells] = x[1, :, cells]
                    return (x,)
                handle = model.model.layers[i].mlp.down_proj.register_forward_pre_hook(hook)
            h = model.model.layers[i](h, attention_mask=mask, position_ids=pos, position_embeddings=rope, use_cache=False)
            if handle:
                handle.remove()
            model.model.layers[i].to('meta')
        last = model.model.norm(h[0, len(seq_a) - 1])
        return torch.nn.functional.linear(last, head).float().cpu()