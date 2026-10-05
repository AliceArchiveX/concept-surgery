# records 目录说明（实验原始读数）

本目录存放全部实验的原始记录。每个 JSON 对应开场白文档中的一个实验环节。**发布脚本重跑后，你的读数应与这些文件一致：答案级（argmax 与目标词排名）在所有环境一致；logits 的逐位一致仅在同一个进程内成立**（跨进程存在与答案无关的小幅数值抖动，见 run_deepseek_direct_replication 的 restored_check 字段）。长文本生成的连接段措辞以本目录存档为准。

## 文件对应关系

| 文件 | 对应实验 | 开场白位置 |
|---|---|---|
| run00_05b_failure_evidence.json | 0.5B 失败证据：基线读数显示它不会 5 的算术（five plus one 排名 17 等），手术无效 | 对照段 |
| run01_arithmetic_flip.json | 六道算术全部翻转（5+2→6 等）+ 三重对照 | 第一环 |
| run01_settings.json | 第一环的固定设置（种子、对照、条件） | 第一环 |
| run02_story_probe_null.json | 0.5B 失败记录 + 故事探针格式教训 | 对照段 |
| run03_story_arithmetic.json | 猴子故事内 5*5→16 方向翻转 | 第一环 |
| run03_monkey_v2.json | 猴子题 v2（新题面，正确答案 25）：正常答 25 / 手术中复述即变"生4只"且答 16 / 两个状态被错误指正（30）时逐字相同地投降 | 第一环 |
| run03_challenge.json | "答案 25 还是 16"挑战探针 | 第一环（埋钩子的异常） |
| run03_identity_collapse.json | "5=4 答 Yes"、"5−4 答 zero"、序列保持 | 第一环 |
| run03_25_meaning.json | 25 仍是 25 的判别实验（第三环核心） | 第三环 |
| run_timeline_midconversation.json | 三→二时间线：否认三十秒前的自己 | 第二环 |
| run_confrontation.json | 五问连爆（单步版。注：step2 的 top5 字段在原始运行中与 argmax 矛盾，已置空并附修正注记；该步以 argmax 及自由生成版为准） | 第三环 |
| run_confrontation_freerun.json | 五问连爆自由生成逐字版：16 与四的幂链、"四四十六不是25"、问号原句、数字四的哲学 | 第三环 |
| run_introspection.json | 两个相反断言都说 Yes + 开放描述七问 + Twenty 记忆消失 | 最后一问 |
| run_deepseek_0731_shared_probe.json | 304B 首次探针：整支共享专家分支替换的初步验证 | 一个更大的脑子 |
| run_deepseek_sparse_flip.json | 304B 稀疏手术（每层 128 格×29 层）：六道算术翻转 + 随机对照 + 恢复 | 一个更大的脑子 |
| run_deepseek_direct_replication.json | 304B 行为五问完整复刻：正常/手术/恢复三态全部逐 token 记录 | 一个更大的脑子 |

## 对照实验记录在哪里

对照不是单独文件，嵌入在每个 JSON 的 results 内：

- `rand128`：随机 128 单元做同样操作，全部读数应与 baseline 一致——这是"手术特异性"的证据；
- `restored`：撤手术后的读数，应与 baseline 逐位一致——这是可逆性的证据；
- `ctrl_rows` / 无关对照题：不含"五"的算术（1+1、3*3 等），全程保持。

## 复现入口

- 五/四手术单题复现（Qwen2.5-32B）：`surgery_switch.py --swap five_four --on/--off --prompt "..."`
- 三/二手术时间线：同上，`--swap three_two`
- 第三环五问连爆（Qwen）：`confrontation_switch.py`（自动跑全部步骤）
- DeepSeek-V4-Flash-0731：`deepseek_switch.py --model <模型目录> --swap five_four --on/--off --prompt "..."`（`--raw` 复现算术卡，`--generate N` 复现行为五问，`--random` 跑随机对照）

每个脚本运行前从 `tables.json` / `deepseek_tables.json` 读取单元表，运行后自动移除全部钩子，磁盘权重不动。

## 关于单元表的说明

tables.json（Qwen，两组）与 deepseek_tables.json（DeepSeek-V4-Flash-0731，一组）里的数字（每层 128 个单元编号）直接使用即可：换上它运行手术，换下它运行对照——发布脚本的全部读数都以这些表为准。