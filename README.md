# Where Does an AI's Understanding Live? We Found "Five", Replaced It, and It Never Noticed.

**AI 把 5 当成了 4 —— 在开源大模型内部做的一次概念置换手术 / Concept-replacement surgery inside open LLMs.**

We located the internal units that carry the identity of the number "five" in Qwen2.5-32B (32B parameters, open weights), and swapped their activations for the ones the model produces when reading "four" — at 128 units per layer, runtime only, weights untouched.

The result: the model still reads "5", still writes "5" when reciting, still knows 5 comes after 4 — but computes with it as four.

```
5+2=     7  →  6          (computes 4+2)
5×5=    25  →  16         (computes 4×4)
5−4=     one  →  zero     (computes 4−4)
5=4?     No   →  Yes      (identity collapse)
2+3=     5   →  5         (boundary case, unchanged - recorded as-is)
3+2=     5   →  5         (no "five" in the prompt; nothing to swap)
```

Asked to re-calculate, it answers *"4×4=16, not 25"* — openly computing in four while you ask it about five. Asked whether it has been tampered with, it says yes to one phrasing and no to the opposite phrasing (both recorded). It can never notice the swap itself: the machinery it would use to check "five" is exactly the machinery that was replaced.

The same surgery was replicated on DeepSeek-V4-Flash-0731 (304B, MoE architecture, intervention on the always-active shared experts) — with its own behavioral signature: arithmetic flips (5−4→zero), direct judgment survives (5=4→"no"), and under pressure to recalculate it drifts into JavaScript.

**Everything here is reproducible on a laptop.** No lab, no cluster: an 8GB-VRAM machine runs the full 32B probe suite; the scripts, the unit tables, and every raw record are in this repo.

## Contents

- **[开场白.md](开场白.md)** — full narrative (Chinese), all experiments in reading order
- **[release/](release/)** — all code, unit tables, probe cards, download guide, raw records:
  - `surgery_switch.py` — the 32B five/four and three/two switches (`--on` / `--off` / `--random` control)
  - `deepseek_switch.py` — the 304B switch
  - `confrontation_switch.py` — the five-question confrontation timeline
  - `tables.json` / `deepseek_tables.json` — the hardcoded unit tables (which neurons, which layers)
  - `probe_cards.md` — every question with expected readings, including boundary cases
  - `records/` — raw JSON of every experiment, mapped one-to-one to the narrative
  - `download.md` — where to get the official models, hardware requirements, how to verify weights by hash

## Quick verify (32B, ~2 min per reading)

```bash
# 1. Download official Qwen2.5-32B (see release/download.md)

# 2. Surgery ON: 5+2 should compute as 4+2
python release/surgery_switch.py --swap five_four --on --prompt "5+2="

# 3. Random-unit control: should change nothing
python release/surgery_switch.py --swap five_four --on --random --prompt "5+2="

# 4. Surgery OFF: original readings restored
python release/surgery_switch.py --swap five_four --off --prompt "5+2="
```

## Reproducibility statement

- Greedy decoding (temperature=0) throughout: same input, same output, every run.
- **Answer-level** readings (argmax, target ranks) reproduce across environments.
- **Logit-level** bitwise identity holds within one process; cross-process runs show small numeric drift that does not change any answer (documented in the records).
- Model weights are never modified — verify by SHA256 against the official release at any time.
- Every claim in the narrative maps to a specific record file; controls (random units, restoration, unrelated arithmetic) are embedded in each record.

## How were the 128 units per layer found?

They are provided as hardcoded tables in this repo, and we use them exactly as provided. The narrative of how they were located is not part of this release.

---

*The model that answered every question in this repo believes, at this moment, that five is four. It has no way to notice. The switch is off now.*
