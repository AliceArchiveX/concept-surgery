"""Surgery switch for Qwen2.5-32B: five->four and three->two activation replacement.

This is the RELEASE version. It contains:
  - the unit tables (per-layer cell indices) for the two swaps we ran
  - the on/off switches for each swap
  - single-prompt reading in both states (normal / surgery on)
  - a random-unit control (--random) for specificity checks
It does NOT contain:
  - how the tables were found (exploration process)
  - any machinery for building tables for new concept pairs

The tables below are plain numbers. We scanned tens of thousands of
units per layer to locate these; why exactly these cells carry the
number-identity signal is not fully understood by us either.

Usage (single-prompt readouts, ~30s each on 8GB-VRAM laptop):
  python surgery_switch.py --swap five_four --off --prompt "5+2="
  python surgery_switch.py --swap five_four --on  --prompt "5+2="
  python surgery_switch.py --swap five_four --on --random --prompt "5+2="
  (donor row is built automatically: same prompt with 5->4 substituted)

Requires: Qwen2.5-32B q8 pack (see download.md), PyTorch + transformers.
Hardware: 64GB RAM + 8GB VRAM works (layer-streamed, ~30s per read).
"""
import argparse
import json
from pathlib import Path

import torch

TABLES = {}


def load_tables():
    global TABLES
    t = json.loads((Path(__file__).parent / 'tables.json').read_text())
    TABLES['five_four'] = t['five_four']
    TABLES['three_two'] = t['three_two']


def donor_of(swap, prompt):
    a, b = ('5', '4') if swap == 'five_four' else ('3', '2')
    return prompt.replace(a, b)


def read_cells(cells_per_layer, mode, inter):
    """Return the per-layer cell list for this run.

    mode 'table'  -> the published unit table (the surgery)
    mode 'random' -> a seeded random selection of the same size (control),
                     drawn from the full intermediate width.
    """
    if mode == 'table':
        return cells_per_layer
    rng = torch.Generator().manual_seed(42)
    return [torch.randperm(inter, generator=rng)[:len(cells_per_layer[i])].tolist()
            for i in range(len(cells_per_layer))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--swap', choices=['five_four', 'three_two'])
    parser.add_argument('--on', action='store_true')
    parser.add_argument('--off', action='store_true')
    parser.add_argument('--random', action='store_true',
                        help='control: random cells instead of the table (surgery-specificity check)')
    parser.add_argument('--model', type=str, default=None,
                        help='path to the q8 pack (default: release/q8_pack)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='prompt to read; prints top-5 and expected-token ranks')
    args = parser.parse_args()
    load_tables()
    if not (args.on or args.off):
        parser.error('specify --on or --off')
    if args.random and not args.on:
        parser.error('--random needs --on')

    from q8_runtime import load_model, forward_single
    model, tok, head, emb = load_model(qroot=args.model)

    if args.prompt:
        inter = model.config.intermediate_size
        table = read_cells(TABLES[args.swap], 'random' if args.random else 'table', inter)
        donor = donor_of(args.swap, args.prompt) if args.on else None
        z = forward_single(model, tok, head, emb, args.prompt,
                           swap=table if args.on else None, donor_prompt=donor)
        print(f'swap={args.swap} on={args.on} cells={"random" if args.random else "table"}')
        print('top5:', [(tok.decode([t]), round(float(v), 2)) for t, v in
                        zip(z.topk(5).indices, z.topk(5).values)])
        print('weights untouched on disk; hooks removed at exit')


if __name__ == '__main__':
    main()
