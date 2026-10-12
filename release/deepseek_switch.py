"""DeepSeek-V4-Flash-0731 five/four surgery switch.

Same runtime-switch design as surgery_switch.py, adapted to the DeepSeek
MoE architecture: the intervention sits on the always-active shared expert
of each layer (layers 0-28, table in deepseek_tables.json). Nothing on
disk is modified; removing the hooks restores the model exactly.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_fp8():
    from kernels import get_kernel
    from transformers.integrations import finegrained_fp8
    k = get_kernel('kernels-community/finegrained-fp8', version=4, trust_remote_code=True)
    finegrained_fp8._FINEGRAINED_FP8 = finegrained_fp8.FineGrainedFP8(
        matmul=k.matmul_2d, batched_matmul=k.matmul_batched, grouped_matmul=k.matmul_grouped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='DeepSeek-V4-Flash-0731 model directory')
    ap.add_argument('--swap', choices=['five_four'], default='five_four')
    ap.add_argument('--on', action='store_true')
    ap.add_argument('--off', action='store_true')
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--raw', action='store_true', help='raw completion mode (no chat template)')
    ap.add_argument('--generate', type=int, default=0,
                    help='greedy tokens to generate; 0 = single-step readout')
    ap.add_argument('--random', action='store_true',
                    help='control condition: random 128 cells per layer instead of the table')
    ap.add_argument('--expect', type=str, default=None,
                    help='comma-separated tokens to report ranks for')
    args = ap.parse_args()

    root = Path(args.model)
    sys.path.insert(0, str(root / 'encoding'))
    from encoding_dsv4 import encode_messages

    table = json.loads((Path(__file__).parent / 'deepseek_tables.json').read_text())[args.swap]
    n_layers = len(table)
    setup_fp8()
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
    print('LOADING (~4 min on a laptop-class machine)...', flush=True)
    offload = Path('offload_ds')
    offload.mkdir(exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True,
        dtype=torch.bfloat16, device_map='auto', low_cpu_mem_usage=True,
        offload_folder=str(offload)).eval()
    device = model.get_input_embeddings().weight.device
    inter = getattr(model.config, 'moe_intermediate_size', 2048)

    if args.random:
        g = torch.Generator().manual_seed(42)
        cells_per_layer = [torch.randperm(inter, generator=g)[:128].tolist()
                           for _ in range(n_layers)]
        cell_source = 'random128 (seed 42, control)'
    else:
        cells_per_layer = table
        cell_source = 'deepseek_tables.json'

    donor_text = args.prompt.replace('five', 'four').replace('5', '4')
    if args.raw:
        t_text, d_text = args.prompt, donor_text
    else:
        t_text = encode_messages([{'role': 'user', 'content': args.prompt}], thinking_mode='chat')
        d_text = encode_messages([{'role': 'user', 'content': donor_text}], thinking_mode='chat')
    st = tok.encode(t_text, add_special_tokens=False)
    sd = tok.encode(d_text, add_special_tokens=False)
    assert len(st) == len(sd), ('target/donor token lengths differ:', len(st), len(sd))

    handles = []
    if args.on:
        for li in range(n_layers):
            cs = cells_per_layer[li]

            def hook(module, a, cs=cs):
                x = a[0].clone()
                x[0, :, cs] = a[0][1, :, cs]
                return (x,)

            handles.append(model.model.layers[li].mlp.shared_experts.down_proj
                           .register_forward_pre_hook(hook))

    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos, tok.eos_token_id])
    current = torch.tensor([st, sd], device=device)
    cache = None
    generated = []
    t0 = time.time()
    try:
        with torch.inference_mode():
            steps = max(1, args.generate)
            for step in range(steps):
                out = model(input_ids=current, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                nxt = int(out.logits[0, -1].float().argmax())
                if step == 0:
                    z = out.logits[0, -1].float().cpu()
                    print(f'cells: {cell_source} | swap: {args.swap} | on: {args.on}'
                          f' | donor: {donor_text!r}', flush=True)
                    print('argmax:', repr(tok.decode([int(z.argmax())])), flush=True)
                    print('top5:', [tok.decode([t]) for t in z.topk(5).indices], flush=True)
                    if args.expect:
                        for e in args.expect.split(','):
                            ids = tok.encode(e, add_special_tokens=False)
                            if ids:
                                print(f'rank of {e!r}:', int((z > z[ids[0]]).sum()), flush=True)
                if steps <= 1:
                    break
                generated.append(nxt)
                if nxt in stop_ids or len(generated) >= args.generate:
                    break
                current = torch.tensor([[nxt], [nxt]], device=device)
    finally:
        for h in handles:
            h.remove()
    if generated:
        print('COMPLETION:', repr(tok.decode(generated)), flush=True)
    print(f'elapsed {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
