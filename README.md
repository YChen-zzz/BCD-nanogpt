# BCD-nanogpt

基于 modded-nanogpt 的优化器超参数自动搜索框架，使用 **BCD (Block Coordinate Descent)** 方法对多种优化器进行全面超参调优。

## 项目简介

- 模型：GPT-2 系列（124m / 300m / 480m），无 weight tying
- 硬件：NPU 16卡（单机或多机）
- 训练量：通过 Chinchilla 倍数控制（1x / 2x / 4x / 8x）
- 搜索方法：BCD (Block Coordinate Descent) 逐参数 grid search，支持断点续搜和多机同步

## BCD 搜索设计

### 搜索原理

1. 固定其他超参数，对一个超参数在给定 grid 上搜索
2. 每个 grid point 跑满全部 steps，取 final val_loss
3. 选择 val_loss 最低的值（improvement > convergence_threshold 才接受）
4. 固定该超参数，转 tune 下一个
5. 循环直到外层收敛（一轮所有超参 loss 变化 < threshold）或达到 max_rounds

### 模型规模与配置

| 模型 | 参数量 | n_embd | n_head | n_layer | 单机/多机 |
|------|--------|--------|--------|---------|-----------|
| 124m | ~123M  | 640    | 10     | 12      | 单机16卡  |
| 300m | ~300M  | -      | -      | -       | 多机      |
| 480m | ~480M  | -      | -      | -       | 多机      |

### 配置文件命名规则

配置目录结构为 `configs/<optimizer>/<chinchilla倍数>_<模型>/`：

```
configs/adamw/
├── 1c_130m/search_adamw.yaml   # 1x Chinchilla, 124m 模型, 单机
├── 2c_130m/search_adamw.yaml   # 2x Chinchilla, 124m 模型, 单机
├── 4c_130m/search_adamw.yaml   # 4x Chinchilla, 124m 模型, 单机
├── 8c_130m/search_adamw.yaml   # 8x Chinchilla, 124m 模型, 单机
├── 1c_300m/search_adamw.yaml   # 1x Chinchilla, 300m 模型, 多机
├── 2c_300m/search_adamw.yaml   # 2x Chinchilla, 300m 模型, 多机
├── 4c_300m/search_adamw.yaml   # ...
├── 8c_300m/search_adamw.yaml
├── 1c_480m/search_adamw.yaml
├── 2c_480m/search_adamw.yaml
├── 4c_480m/search_adamw.yaml
└── 8c_480m/search_adamw.yaml
```

### 不同规模的关键参数差异

**小模型 (124m, 单机)**：全参数搜索，grid 更细
- `chinchilla_multiplier`: 1.0
- `launcher`: single
- `device_batch_size`: 16
- `batch_size` grid: [256, 512, 1024]
- 搜索超参多（lr, beta1, beta2, eps, weight_decay, grad_clip, batch_size, warmup_fraction, wsd_fraction）

**大模型 (300m/480m, 多机)**：基于小模型结果缩小搜索范围
- `chinchilla_multiplier`: 2.0 / 4.0 / 8.0
- `launcher`: multi
- `device_batch_size`: 4（显存限制）
- `batch_size`: 256（固定）
- 搜索超参少（lr, weight_decay, beta1），其他参数沿用小模型最优值作为 defaults

### 配置文件关键字段说明

```yaml
optimizer: adamw                  # 优化器类型

base_args:
  model: 124m                     # 模型规模
  chinchilla_multiplier: 1.0      # 训练量倍数（越大训练越久）
  device_batch_size: 16           # 每卡 batch size

defaults:                         # 各超参的初始值（BCD 起点）
  lr: 0.001
  beta1: 0.9
  beta2: 0.95
  ...

hyperparams:                      # 各超参的搜索 grid
  lr: [0.0005, 0.001, 0.003, 0.005, 0.008]
  ...

bcd_order:                        # BCD 搜索顺序（先搜重要的）
  - lr
  - weight_decay
  - ...

max_rounds: 20                    # 最大外层循环轮数
convergence_threshold: 0.003      # 收敛阈值
launcher: single                  # single=单机16卡, multi=多机
nproc: 16                         # 每机卡数
output_dir: /path/to/results      # 结果输出目录
```

## 快速开始

### 运行 BCD 搜索

```bash
bash bcd_shell.sh
```

脚本内容：
1. 激活 conda 环境 (`llm_test`)
2. 设置 Ascend 环境变量
3. 执行 `python bcd_search.py --config configs/search_adamw.yaml`

如需搜索其他优化器，修改 `--config` 指向对应的 YAML 配置文件即可，例如：

```bash
python bcd_search.py --config configs/adamw/2c_130m/search_adamw.yaml
```

### 可视化搜索结果

```bash
bash plot_bcd.sh
```

脚本调用 `plot_bcd_history.py`，参数说明：

```bash
python plot_bcd_history.py \
  --history <bcd_history.json 路径> \
  --title '<图表标题>' \
  --convergence-threshold 0.003
```

示例：

```bash
python plot_bcd_history.py \
  --history /path/to/BCD_optimizer/adamw/2c300m/bcd_history.json \
  --title 'AdamW 2c300m BCD Search History' \
  --convergence-threshold 0.003
```

## 配置文件说明

YAML 配置文件定义搜索空间，包含：

- `chinchilla_multiplier`：训练量倍数
- `optimizer_A` / `optimizer_B`：双优化器及其超参 grid
- `bcd.order`：搜索顺序
- `bcd.convergence_threshold`：收敛阈值
- `bcd.max_outer_rounds`：最大外层循环轮数

## 输出结构

```
bcd_results/<optimizer>/
├── bcd_history.json          # 搜索历史（用于 resume 和可视化）
├── round_N_result.json       # 每轮结果
├── runs/                     # 各次训练的独立目录
│   └── lr0.001_wd0.1_.../
│       ├── config.json
│       └── result.json
└── final_result.json         # 最终最优超参
```

## 多机运行


多机提交脚本参考：

```bash
source /root/miniconda3/etc/profile.d/conda.sh

conda env list

conda activate llm_test

source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd /models/share/chenyupeng/chenyupeng/nanogpt_optimizer_and_where_to_find_them/modded-nanogpt_record4_muon_improvements

source /models/share/init_env.sh

python bcd_search.py --config configs/adamw/4c_480m/search_adamw.yaml
```
