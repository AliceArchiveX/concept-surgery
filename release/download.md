# 模型下载说明

## 你需要下载什么

**只有一个文件包需要下载：官方 Qwen2.5-32B 模型。**我们发布的所有实验代码都在这个官方模型上运行，发布包里不含任何修改过的模型。

### 下载地址（官方渠道）

- ModelScope（国内）：`Qwen/Qwen2.5-32B`
- Hugging Face（国际）：`Qwen/Qwen2.5-32B`

两处的模型完全一致。约 61GB（17 个 safetensors 分片）。

### 校验

下载完成后，请核对分片的 SHA256 与官方发布页一致。这确认两件事：
1. 你拿到的是官方模型，没有被任何中间环节篡改；
2. 我们全部实验读数所用的模型，与你手里的每一个字节相同。

## q8 权重包（可选但推荐）

直接跑 BF16 原始权重，每个读数约 60 秒。为了让你在普通电脑上跑得动（8GB 显存），我们实验后期把同一模型的权重转成 int8 存储（磁盘体积 62GB→34GB，数值误差每层最大 0.003，全部考卷读数与 BF16 一致——用包内 `q8_verify.py` 可自行验证）。

转换脚本 `make_q8.py` 附在发布包里：**输入官方模型目录，输出 q8 包。**你也可以跳过转换直接跑 BF16，只是慢一倍。

需要明确的是：q8 转换是纯数值格式变换，不涉及任何实验干预。干预（手术）永远是运行时的，两个包的磁盘权重都是"官方内容"。转换与验证脚本都附在包里：`make_q8.py --model <官方模型目录> --out q8_pack/` 生成 q8 包；`q8_verify.py --model <q8包目录>` 跑五道探针，读数应与 records 一致（约每道 30 秒）。

## 硬件要求

| 配置 | BF16 原始 | q8 包 |
|---|---|---|
| 显存 | 8GB | 8GB |
| 内存 | 64GB | 64GB |
| 磁盘 | 61GB | 34GB |
| 单次读数 | ~60 秒 | ~30 秒 |

无显卡也能跑（纯 CPU 流式，约 5 分钟一次读数），代码自动检测。

## 运行

```bash
# 下载官方模型后：
python make_q8.py --model <官方模型目录> --out q8_pack/

# 复现第一张考卷（五/四手术，开）：
python surgery_switch.py --swap five_four --on --prompt "5+2="

# 撤销手术，确认恢复：
python surgery_switch.py --swap five_four --off --prompt "5+2="
```

每次运行都会打印 top5 候选和目标词排名，与 probe_cards.md 的预期逐条对照。

## DeepSeek-V4-Flash-0731（可选，304B 复现）

开场白"一个更大的脑子"一节的全部读数，用这个模型复现。

**下载：** ModelScope / Hugging Face 官方 `DeepSeek-V4-Flash-0731`（约 167GB，48 个分片）。下载后按官方发布页核对分片校验值。注意：该仓库不带 `tokenizer.json`，从官方 `DeepSeek-V4-Flash`（284B 预览版）仓库单独下载 `tokenizer.json` 放进模型目录即可。

**额外依赖：** 该模型注意力为 FP8 实现，需要 `kernels` 与 `triton-windows`（首次运行会联网拉取 `kernels-community/finegrained-fp8` 内核，之后走本地缓存），以及 `accelerate`。

**硬件实况（我们跑这套的机器）：** 8GB 显存的笔记本 + 64GB 内存。模型大部分常驻内存/磁盘，单步读数约 2 分钟，逐 token 生成约 2 分钟一个字——慢，但完全可跑。跑不动 304B 不影响其余全部内容的复现。

```bash
# 算术卡（单步读数）：
python deepseek_switch.py --model <模型目录> --swap five_four --on --raw --prompt "5+2="

# 行为五问（对话模式，生成 16 个 token 起步）：
python deepseek_switch.py --model <模型目录> --swap five_four --on --generate 16 --prompt "5减4等于几？"

# 随机对照（应全部保持原读数）：
python deepseek_switch.py --model <模型目录> --swap five_four --on --random --raw --prompt "5+2="
```