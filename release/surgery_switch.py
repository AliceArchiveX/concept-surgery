"""Surgery switch for Qwen2.5-32B: five->four and three->two activation replacement.

This is the RELEASE version. It contains:
  - the unit tables (per-layer cell indices) for the two swaps we ran
  - the on/off switches for each swap
  - the probe prompts used in the published timeline
It does NOT contain:
  - how the tables were found (exploration process)
  - any machinery for building tables for new concept pairs

The tables below are plain numbers. We scanned tens of thousands of
units per layer to locate these; why exactly these cells carry the
number-identity signal is not fully understood by us either.

Usage:
  python surgery_switch.py --swap five_four --on
  python surgery_switch.py --swap five_four --off
  then run any probe prompt through the model.

Requires: Qwen2.5-32B from official channels (see download.md),
the q8 pack from this release, PyTorch + transformers.
Hardware: 64GB RAM + 8GB VRAM works (layer-streamed, ~30s per read).
"""
import argparse
import json
from pathlib import Path

import torch

# Per-layer unit indices for the two published swaps.
# Format: FIVE_FOUR[layer] = [cell indices], 64 layers, 128 cells each.
FIVE_FOUR = None  # loaded from tables.json (keeps this file readable)
THREE_TWO = None

TABLES = {}


def load_tables():
    global TABLES
    t = json.loads((Path(__file__).parent / 'tables.json').read_text())
    TABLES['five_four'] = t['five_four']
    TABLES['three_two'] = t['three_two']


def build_hooks(model, swap, on):
    """Attach or remove the replacement hooks for one swap."""
    if not on:
        return []
    cells_per_layer = TABLES[swap]
    handles = []
    for i, layer in enumerate(model.model.layers):
        cells = cells_per_layer[i]

        def hook(module, args, cells=cells):
            x = args[0].clone()
            x[0, :, cells] = x[1, :, cells]
            return (x,)
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    return handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--swap', choices=['five_four', 'three_two'])
    parser.add_argument('--on', action='store_true')
    parser.add_argument('--off', action='store_true')
    parser.add_argument('--prompt', type=str, default=None,
                        help='optional: run this prompt and print the top-5')
    args = parser.parse_args()
    load_tables()
    if not (args.on or args.off):
        parser.error('specify --on or --off')

    # Model loading follows the q8 streamed pattern from the release notes.
    from q8_runtime import load_model, forward_single
    model, tok, head, emb = load_model()
    handles = build_hooks(model, args.swap, args.on)

    if args.prompt:
        z = forward_single(model, tok, head, emb, args.prompt)
        print('top5:', [(tok.decode([t]), round(float(v), 2)) for t, v in
                        zip(z.topk(5).indices, z.topk(5).values)])

    for h in handles:
        h.remove()
    print(f'swap {args.swap} {"attached" if args.on else "removed"}; '
          f'weights untouched on disk')


if __name__ == '__main__':
    main()