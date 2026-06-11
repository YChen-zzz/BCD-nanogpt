# Task: Optimizer Hyperparameter BCD Search Framework

## 项目背景

- 已有一个能在 NPU (16卡) 上跑通的 modded-nanogpt 预训练代码
- 模型（改造后）: GPT-2 ~123M (n_layer=12, **n_head=10**, **n_embd=640**, 无 weight tying)
- 模型（改造前）: GPT-2 ~124M (n_layer=12, n_head=12, n_embd=768, 有 weight tying)
- 旧 Baseline: Muon + AdamW, val_loss = 3.2779 (n_embd=768, weight tying, 6200 steps)
- 每 step 524288 tokens (batch_size=512, seq_len=1024)
- Chinchilla 1x = 2.47B tokens = ~4704 steps（基于 123.3M 参数）

## 目标

构建一个**一键启动的自动化超参搜索框架**，用于对 10+ 个优化器进行 fully tune，搜索方法为 **BCD (Block Coordinate Descent)**。

## 核心设计决策

### 1. 优化器架构：双优化器方案（分开 BCD）

当前 codebase 采用**双优化器**架构：
- **Optimizer A**: 作用于 `transformer.h`（transformer blocks）
- **Optimizer B**: 作用于 `lm_head / wte`（embedding/output head）

BCD 搜索策略：**先 tune 完一个 optimizer 的所有超参，再 tune 另一个 optimizer**（而非交替搜索），因为两个 optimizer 作用于不同参数组，分开搜更清晰。

参考实现: `/mnt/host-model/chenyupeng/nanogpt_shipei/modded-nanogpt-Adam_tuned_LR/`（纯 Adam 版本）

### 2. 训练量控制：Chinchilla 倍数

- 用户输入一个 **Chinchilla 倍数**（如 1x, 1.5x, 2x）
- 系统自动计算所需 total tokens = 倍数 × 123.3M × 20
  - 1x Chinchilla = 2.47B tokens = ~4704 steps
- **所有 optimizer 和所有 run 使用相同 total tokens & dataloader 顺序**（保证公平对比）
- num_iterations = total_tokens / tokens_per_step (524288)

### 3. BCD 搜索方法

1. 固定其他超参数，对一个超参数在给定 grid 上搜索
2. 每个 grid point 跑满全部 steps，取 final val_loss
3. 选择 val_loss 最低的值作为该超参数的最优值
4. 固定该超参数，转 tune 下一个
5. 循环直到外层收敛（一轮所有超参 loss 变化 < threshold）

**搜索顺序固定**（可在 config 中指定，不做动态调整）

### 4. Seed & 收敛阈值

- **需要先实现**: 固定 random seed (torch, numpy, data loader)
- **需要先验证**: 相同配置跑 2-3 次的 val_loss variance
- 收敛阈值暂定 3e-3，待验证后确定

### 5. 模型配置变更：n_embd=640, 移除 Weight Tying

**移除 weight tying**：
- 当前代码使用 `self.transformer.wte.weight = self.lm_head.weight`
- **决策：移除**，wte 和 lm_head 独立参数，避免实验结论受 weight tying 影响

**缩小 n_embd 至 640** 以保持 ~124M 参数量：
- n_embd: 768 → 640
- n_head: 12 → 10 (head_dim=64)
- 总参数量: 123.3M（与原 124M 几乎一致）
- 好处：保持和原始 GPT-2 124M 相同量级的参数，Chinchilla 倍数计算也保持一致

### 6. QKV 分开 vs 合并

当前 record4 代码：`c_attn = nn.Linear(n_embd, 3*n_embd)` （QKV 合并为一个 Linear）
record7 参考代码：`c_q`, `c_k`, `c_v` 分开为三个 Linear

**对 optimizer tuning 的影响**：
- Muon 在 record4 中对 QKV 合并矩阵做特殊处理：`if g.size(0) == 3 * g.size(1)` 时 split 后分别做 orthogonalization
- record7 分开后，每个 linear 独立处理，更干净
- **建议分开**：分开后 Muon 的 code path 更简洁，且对于其他 optimizer（如 Adam）没有 QKV split 的 special case
- 分开不会改变参数量，只是组织方式不同

### 7. Muon Distributed 方法

record7 的 Muon 使用了**分布式 orthogonalization**：
- 将参数按 `i % world_size == rank` 分配到不同 GPU
- 每个 GPU 只对自己负责的参数做 Newton-Schulz
- 然后通过 `dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)` 同步

record4 的 Muon 是**每个 GPU 独立对所有参数做完整 Newton-Schulz**。

**是否会改变训练 dynamics？**
- **不会**。distributed 版本只是把计算分摊到不同 GPU 上，最终 all_reduce SUM 后每个 GPU 拿到的 update 是完全一样的。数学上等价。
- 好处：减少每个 GPU 的计算量（加速），尤其当 layers 多时
- **建议加入**：12 层 transformer 刚好可以被 16 GPU 整除，效率更高，且数学等价不影响结果

### 8. 精度策略

**目标**：activation 为 bf16，其他（权重、optimizer state、梯度）均为 fp32。

**当前代码精度审计** (record4)：
- ✅ `torch.amp.autocast(device_type='npu', dtype=torch.bfloat16)` — forward/backward activation 为 bf16
- ✅ 模型权重默认 fp32（`nn.Linear`, `nn.Embedding` 默认 float32）
- ✅ `logits = logits.float()` — loss 计算在 fp32
- ✅ `rmsnorm` 内部 `x = x0.float()` — 归一化在 fp32
- ⚠️ **Muon Newton-Schulz 在 bf16 做** (`X = G.bfloat16()`)，但最终 `return X.to(G.dtype)` 转回 fp32 再 apply
- ⚠️ **Rotary embedding 的 cos/sin cached 为 bf16** (`freqs.cos().bfloat16()`)
- ✅ Optimizer state (momentum_buffer): `torch.zeros_like(g)` 跟 grad 同 dtype，grad 在 autocast 下应为 fp32（因为 master weight 是 fp32）

**结论**：大体满足要求。Newton-Schulz 内部使用 bf16 是 Muon 的设计意图（论文中说明可以稳定运行在 bf16），这是 optimizer 内部的计算精度选择，不影响"权重/state 为 fp32"的目标。Rotary cos/sin 为 bf16 因为它参与 activation 计算（在 autocast 范围内），也是合理的。

**如需严格 fp32**（连 Newton-Schulz 都用 fp32），需要移除 `G.bfloat16()` 这一步，但会显著增加 Muon 计算量且偏离原始设计。建议保持现状。

## 超参数空间（以 Muon+AdamW 为例）

```yaml
chinchilla_multiplier: 1.31

optimizer_A:  # transformer.h
  type: muon
  hyperparams:
    lr: [0.0002, 0.00036, 0.0005, 0.0007, 0.001]
    momentum: [0.9, 0.95, 0.99]

optimizer_B:  # lm_head
  type: adamw
  hyperparams:
    lr: [0.001, 0.002, 0.0036, 0.005, 0.008]
    beta1: [0.85, 0.9, 0.95]
    beta2: [0.95, 0.99, 0.999]
    epsilon: [1e-8, 1e-7, 1e-6]
    weight_decay: [0, 0.01, 0.1]

scheduler:  # 共享 scheduler
  warmup_iters: [0, 100, 200, 400]
  warmdown_fraction: [0.2, 0.29, 0.35, 0.4]

bcd:
  order: [optimizer_A.lr, optimizer_B.lr, optimizer_B.weight_decay, optimizer_B.beta1, optimizer_B.beta2, optimizer_A.momentum, scheduler.warmup_iters, scheduler.warmdown_fraction, optimizer_B.epsilon]
  convergence_threshold: 3e-3  # 待确认
  max_outer_rounds: 3
```

## 待确认事项

### P0: Seed 确定性
- [ ] 添加 seed 控制到 train_gpt2.py（torch, numpy, dataloader）
- [ ] 跑 2-3 次相同配置，测量 val_loss variance
- [ ] 根据 variance 确定 BCD convergence_threshold

### P1: 架构变更后验证
- [ ] 移除 weight tying + n_embd 改为 640 + n_head 改为 10
- [ ] 移除 weight tying 后重新跑 baseline，确认新的 val_loss 水平
- [ ] QKV 分开后确认训练 loss 曲线没有 regression
- [ ] Muon distributed 加入后确认结果与非 distributed 版本一致（数学等价验证）

### P2: 细节确认
- [x] BCD 分开还是交替 → **分开（先 A 后 B）**
- [x] BCD 外层最多循环几轮 → **暂定 max 3 轮**
- [x] val_loss_every 在搜索时是否需要调整？（当前每 125 steps eval 一次）→ 保持不变
- [x] 移除 weight tying 后 Chinchilla 倍数基准参数量 → **n_embd=640, ~123.3M**

## 实现计划

### Phase 0: 基础准备 & 代码清理 ✅
- [x] 移除 weight tying（wte 和 lm_head 独立参数）+ n_embd=640, n_head=10
- [x] QKV 分开为 c_q, c_k, c_v（参考 record7）
- [x] 加入 Muon distributed orthogonalization（参考 record7）
- [x] 添加 seed 控制（torch, numpy, cudnn）
- [x] 两阶段 argparse：先解析 --optimizer，再注册对应超参数
- [x] optimizers/ 文件夹：muon_optimizer.py, adamw_optimizer.py
- [x] MODEL_CONFIGS 映射表（--model 124m）

### Phase 1: Optimizer 框架 ✅ (合并到 Phase 0)
- [x] optimizer registry + load_optimizer_module
- [x] 支持 Chinchilla 倍数输入 → 自动计算 num_iterations
- [x] 每次 run 输出 _config.json + _result.json

### Phase 2: BCD 搜索引擎 ✅
- [x] bcd_search.py：读取 YAML → 逐参数搜索 → 收敛判断
- [x] 结果存储：bcd_history.json + round_N_result.json + final_result.json
- [x] 支持断点续搜（跳过已完成的 grid point）

### Phase 3: 一键启动 ✅
- [x] `bash launch_search.sh configs/search_muon.yaml`
- [x] 内部调用 `torchrun ... train_gpt2.py --optimizer muon ...`
- [x] 搜索完成后输出最优超参到 final_result.json

### 待验证
- [ ] 在 NPU 上实际跑通 train_gpt2.py (n_embd=640, 无 weight tying)
- [ ] 固定 seed 后跑 2-3 次确认 val_loss variance → 确定 convergence_threshold

## 时间估算

以 Muon+AdamW 双优化器为例（1x Chinchilla, ~4704 steps, ~31min/run）：
- 超参总数 ~9 个，平均 grid size ~4
- BCD 2 轮: 9 × 4 × 2 = 72 次训练
- 每次 ~31min → **总计 ~37 小时/优化器**
- 10 个优化器并行在多台机器 → 可在 ~37 小时内完成

## 讨论记录

### 2025-XX-XX: Round 1 - 初始需求讨论
- 用户目标：对 10+ 个优化器做 fully hyperparameter tuning via BCD
- 阈值暂定 3e-3，需先确认 seed reproducibility
- 用户希望输入尽量简单（一个 config + 一键启动）

### 2025-XX-XX: Round 2 - 关键决策确认
- **双优化器都要 tune**：两个 optimizer 各自超参独立搜索
- **训练量用 Chinchilla 倍数控制**：用户指定倍数，系统自动算 tokens/steps，保证公平
- **BCD 搜索顺序固定**（在 config 中指定）
- **每次都跑满**，不做短 step 初筛
- 新优化器会从外部 repo 手动/自动引入（如 Adam tuned LR 版本）

### 2025-XX-XX: Round 3 - 代码架构决策
- **BCD 分开 tune**：先 tune 完 optimizer A 的所有超参，再 tune optimizer B（而非交替）
- **移除 weight tying**：wte 和 lm_head 参数独立，避免实验结论受 weight tying 影响
- **QKV 分开**：从 `c_attn(3*n_embd)` 改为 `c_q, c_k, c_v` 三个独立 Linear，简化 Muon code path
- **加入 Muon distributed**：参考 record7，分布式 orthogonalization，数学等价但更高效
  - 多机场景：每个优化器在一台独立 16 卡机器上跑，不同优化器并行，distributed Muon 无影响
  - 注意：如果将来单次 run 跨多机（world_size>16），12 层分给 >16 个 rank 时大部分 rank 空闲，需要调整
- **精度策略确认**：activation bf16 + 权重/state/grad fp32，当前代码基本满足
  - Newton-Schulz 内部 bf16 是 Muon 设计意图，保持
  - Rotary cos/sin bf16 参与 activation 计算，合理
- **Chinchilla 基准用实际参数量**（移除 weight tying 后）
- **多机器并行**：用户有多台 NPU 机器，不同优化器可以并行搜索

### 2025-XX-XX: Round 4 - 参数量 & n_embd 讨论
- 移除 weight tying 后参数量对比：
  - n_embd=768 无 tying → 162.1M (Chinchilla 1x = 3.24B tokens, ~6185 steps)
  - n_embd=640 无 tying → 123.3M (Chinchilla 1x = 2.47B tokens, ~4704 steps) ← 与原 124M 几乎一致
  - n_embd=640 时 n_head 需调整为 10（head_dim=64）
- **待确认**：选 768 (163M) 还是 640 (123M)？→ **已确认：n_embd=640, n_head=10, ~123.3M 参数**

### 2025-XX-XX: Round 5 - BCD 搜索历史记录 & run 管理

**问题：当前 bcd_search.py 的 key 设计不够好**

当前用 `(round, param_name, param_value)` 作为去重 key，存在两个问题：

1. **key 不完整**：只记录了当前搜索的参数名和值，没有记录其他 fixed 超参的值。
   同一个 `(round=2, lr, 4e-3)` 如果其他超参已经变了，实际上是不同的实验。
2. **跨 round 无法去重**：Round 1 跑过 `{lr=4e-3, wd=0, eps=1e-8}`，Round 2 如果
   `wd` 和 `eps` 没有变化，搜 `lr` 时会重复跑完全相同的配置。

**改进方案：用完整超参组合作为去重 key**

- 每次 run 的唯一标识 = 所有超参数值的完整组合（排序后 hash 或 frozen dict）
- `bcd_history.json` 中每条记录包含完整的超参数 dict
- 去重时比较完整超参组合，而非 `(round, param_name, value)`
- 好处：Round 2 搜 `lr` 时，如果其他超参没变，自动复用 Round 1 的结果

示例 key 生成方式：
```python
import hashlib, json
def config_key(params_dict):
    s = json.dumps(params_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]
```

**改进方案：每个 run 独立文件夹**

当前 `train_gpt2.py` 用 uuid 作为 run_id，log 和 ckpt 混在 `logs/` 下。改进：

- 每个 run 生成一个有意义的文件夹名，包含关键超参信息
- 文件夹结构：
  ```
  bcd_results/muon/
  ├── bcd_history.json                    # 全局搜索历史
  ├── round_1_result.json
  ├── runs/
  │   ├── muon_lr=0.0001_adamw_lr=0.0036_..._a1b2c3/   # hash 后缀避免名字冲突
  │   │   ├── config.json                # 完整超参配置
  │   │   ├── result.json                # final_val_loss
  │   │   ├── train.log                  # 训练日志
  │   │   └── state_step004704.pt        # checkpoint (可选)
  │   ├── muon_lr=0.0002_adamw_lr=0.0036_..._c3d4e5/
  │   └── ...
  └── final_result.json
  ```
- `bcd_history.json` 中每条记录增加 `run_dir` 字段指向对应的 run 文件夹
- `train_gpt2.py` 需要支持 `--output_dir` 参数来控制日志/ckpt 存放位置

**待确认**
- [ ] run 文件夹命名格式：纯 hash？关键超参 + hash？
- [ ] checkpoint 是否默认保存？（搜索时可能不需要 ckpt 节省磁盘）
- [ ] `train_gpt2.py` 的 `--output_dir` 参数设计

### 2026-05-26: Round 6 - BCD 搜索语义 & 多机同步改进

**已实现改进**

- **单个 block 内也使用 `convergence_threshold` 控制是否采用新超参**：
  - 旧逻辑：每个参数 grid 搜完后，总是先把 `current_best[param_name]` 改成 grid 内最低 loss 的值；`convergence_threshold` 只控制本轮是否继续。
  - 新逻辑：只有 `old_loss - param_best_loss > convergence_threshold` 时，才更新 `current_best[param_name]`。
  - 如果 improvement 小于等于阈值，保留旧值，避免由于 seed variance 或极小波动频繁切换超参。
- **`best_val_loss` 与 `best_params` 对齐**：
  - 使用 `current_best_loss` 表示当前被 BCD 接受的配置对应 loss。
  - 被阈值拒绝的 candidate loss 不会写成 final 的 best loss，避免 `final_result.json` 里参数和 loss 不对应。
- **baseline-first 搜索顺序**：
  - 每个参数 block 内先评估当前旧值作为 baseline，再遍历 grid 中其它值。
  - 如果 default/current value 不在 grid 中，也会先作为 baseline 评估。
- **history entry 增加比较字段**：
  - `old_param_value`
  - `old_loss`
  - `loss_delta_vs_old`
  - `old_params`
  - 这些字段用于直接查看每个新训练 grid point 相对当前 baseline 的 improvement。
- **history 写入防重复**：
  - `save_history_entry()` 按完整 `all_params` 去重，避免同一完整配置重复追加到 `bcd_history.json`。
  - 注意：当前 `bcd_history.json` 仍然主要承担 run cache 角色；如果同一配置在后续 round 作为 cache hit 出现，不会额外写一条新的 BCD trace。
- **JSON 原子写入**：
  - `bcd_history.json`、`round_N_result.json`、`final_result.json` 使用临时文件 + `os.replace()` 写入。
  - 避免进程中断时留下半截 JSON。
- **dry-run 不写 metadata**：
  - `--dry_run` 只打印训练命令和模拟流程，不写 `bcd_history.json` / `round_N_result.json` / `final_result.json`。
  - 避免 fake `999.0` loss 污染正式 resume。
- **失败处理更严格**：
  - 如果 baseline 配置训练失败或缺少结果，直接报错停止该 BCD 搜索。
  - 不再把 baseline loss 当作 `inf`，避免把任意 candidate 误判成无限 improvement。

**多机 BCD controller/worker 同步**

当前 multi 模式采用轻量 rank0 decision 文件同步，而不是每个节点独立读 history 决策：

- 所有节点仍然都需要启动 `bcd_search.py`。
- 所有节点都按相同 BCD loop 运行，并在每个需要训练的 grid point 上共同启动 multi-node `torchrun`。
- 只有 `CURRENT_NODE_RANK == controller_node_rank` 的节点写搜索 metadata：
  - `bcd_history.json`
  - `round_N_result.json`
  - `final_result.json`
  - `control/session.json`
  - `control/<search_id>/decision_*.json`
- 默认 `controller_node_rank = 0`，可在 config 中显式指定：
  ```yaml
  controller_node_rank: 0
  ```
- 非 0 节点不读取 `bcd_history.json` 来决定 run/skip，而是读取 rank0 写出的 decision 文件。

decision 文件示例：

```json
{
  "task_idx": 1,
  "round": 1,
  "param_name": "lr",
  "param_value": 0.001,
  "action": "run",
  "val_loss": null,
  "all_params": {"lr": 0.001, "weight_decay": 0.1},
  "run_dir": ".../runs/lr0.001_weight_decay0.1",
  "timestamp": "2026-05-26 12:00:00"
}
```

如果 rank0 发现完整配置已经在 history 中完成，则写：

```json
{
  "action": "skip",
  "val_loss": 3.2919
}
```

非 0 节点收到 `skip` 时不会读 history，而是直接使用 decision 中的 `val_loss` 更新本地 BCD 状态，从而避免各节点因为 history 读写时序不同导致 `torchrun` 错位。

**result.json wait/retry**

- `run_training()` 在训练返回后不会立刻判定 `result.json` 缺失。
- 默认等待 `result_json_timeout=300` 秒，每 `result_json_interval=2` 秒轮询一次。
- 这是为了处理 multi-node 训练中只有 rank0 写 `result.json`、共享文件系统同步有延迟的情况。
- 可在 config 中覆盖：
  ```yaml
  result_json_timeout: 600
  result_json_interval: 2
  ```

**multi 模式启动要求**

在启动 `bcd_search.py` 前，外层 shell 必须已经设置：

```bash
CURRENT_NODE_RANK
TOTAL_NODES
NPUS_PER_NODE
MASTER_IP
MASTER_PORT
```

推荐每台机器执行：

```bash
cd /models/share/chenyupeng/chenyupeng/nanogpt_optimizer_and_where_to_find_them/modded-nanogpt_record4_muon_improvements
source /models/share/init_env.sh
python bcd_search.py --config configs/adamw/2c_130m/search_adamw.yaml
```

注意：`build_train_command()` 内部也会 source `/models/share/init_env.sh`，但那是在子 shell 里执行训练命令时才发生。`bcd_search.py` 自己启动时就需要读取 `CURRENT_NODE_RANK`，所以外层 shell 也必须先 source。

**control 目录与 resume**

multi 模式会创建临时同步文件：

```
output_dir/control/session.json
output_dir/control/<search_id>/decision_000001.json
output_dir/control/<search_id>/decision_000002.json
...
```

- `search_id` 当前由 rank0 每次启动自动生成：`YYYYMMDD_HHMMSS_<pid>`。
- 这些文件只用于多机同步，不是 resume 的真实状态。
- 真正用于 resume 的仍然是 `bcd_history.json` 和各 run 目录下的 `result.json`。
- 如果中途断掉并准备重新启动，确认没有旧的 `bcd_search.py` / `torchrun` 仍在运行后，可以清理 control 目录：

```bash
rm -rf /models/share/chenyupeng/chenyupeng/pretraining_record/BCD_optimizer/adamw/2c130m/control
```

然后所有节点重新启动同一 config。rank0 会生成新的 `control/session.json` 和新的 decision 目录，已完成配置会从 `bcd_history.json` 中恢复并 skip。

### 2026-06-07: Round 7 - 已发现 Bug 记录

**Bug 1（已修复）：`bcd_search.py` — `defaults` 中不在 `bcd_order` 的参数未传给 `train_gpt2.py`**

- **根因**：`current_best` 只初始化 `bcd_order` 中的参数，`defaults` 里不在 `bcd_order` 的参数从未进入 `run_params`，`train_gpt2.py` 回落到自身 `argparse` 默认值。
- **受影响配置**：`2c_300m`、`4c_300m`、`8c_300m`、`2c_480m`、`4c_480m`、`8c_480m`（`1c_*` 和 `*c_130m` 不受影响，其 defaults 全部在 bcd_order 中）。
- **实际影响（以 2c_300m 为例）**：
  - `warmup_fraction` 0.05 → 0.0（无 warmup）
  - `grad_clip` 1.0 → 0.0（无梯度裁剪）
  - `batch_size` 256 → 512（**num_iterations 减半，只训练了目标 Chinchilla tokens 的 50%**）
  - `beta2` 0.95 → 0.999
  - `eps` 1e-16 → 1e-8
- **修复**：在 `bcd_search()` 中计算 `fixed_params = {k: v for k, v in defaults.items() if k not in bcd_param_set}`，并在构建 `run_params` 时合并：`run_params = {**fixed_params, **current_best, param_name: value}`。
- **已污染数据**：`2c300m`（29 runs）、`2c480m`（13 runs）、`4c300m`（22 runs）需清理后重跑。

**Bug 2（已修复）：`bcd_search.py` — `old_key` 缺少 `fixed_params` 导致 RuntimeError**

- **根因**：Bug 1 修复后，`run_params` 含 `fixed_params`，但 `old_params = {**current_best}` 不含，导致 `run_key == old_key` 永不成立，`old_loss` 始终 None，训练完成后抛 RuntimeError。
- **修复**：`old_params = {**fixed_params, **current_best}`。

**Bug 3（已修复）：`train_gpt2.py` — `save_every=0` 仍保存末步 checkpoint，磁盘消耗巨大**

- **根因**：条件 `if master_process and (last_step or ...)` 中 `last_step` 不受 `save_every` 控制，每次 BCD run 均保存完整模型权重（130m≈460MB，300m≈1.2GB，480m≈1.8GB）。
- **实测消耗**：已完成的 ~300 次 run 共占用约 243GB。全量搜索预计超过 1TB。
- **修复**：新语义 `save_every=0`=从不保存，`<0`=仅末步，`>0`=每 N 步+末步。

**已知问题（待修复，暂不改以免影响在跑实验）：`train_gpt2.py` — `warmdown_iters` 计算语义有误**

- **根因**：当前 `warmdown_iters = int(wsd_fraction * num_iterations)`，即 wsd_fraction 是占**总步数**的比例。
- **正确语义**：`wsd_fraction` 应是占**非 warmup 步数**的比例，即：
  ```python
  warmdown_iters = int(wsd_fraction * (num_iterations - warmup_iters))
  ```
- **影响**：当 `warmup_fraction + wsd_fraction > 1.0`（如 480m 配置 `warmup=0.05, wsd=1.0`）时，warmup 结束处 LR 从 1.0 突降到 `1 - warmup_fraction = 0.95`，存在 5% 断层。用正确公式则在衔接点精确为 1.0，无断层。
- **为何暂不修**：1c_480m 已在旧公式下完成搜索并确定了最优超参（包括 `wsd_fraction=1.0`）；贸然修改会使后续 2c/4c/8c_480m 与 1c_480m 使用不同 LR 曲线，破坏可比性。待 1c_480m 重新确认后再统一修复。

---

**仍待改进**

- `bcd_history.json` 目前同时承担 run cache 和部分 BCD trace 角色。更干净的长期方案是拆成：
  - `run_results.json`：完整 `all_params -> val_loss/run_dir`，只做 cache/resume。
  - `bcd_history.json`：每一次 BCD 比较事件都 append，允许同一 `all_params` 在不同 baseline 下重复出现。
- `params_to_key()` 目前使用 `str(v)`，如果手动编辑 YAML/JSON 导致 `1` vs `1.0` 或字符串/数值混用，可能产生不同 key。后续可加入 canonical normalize。
- `params_to_dir_name()` 目前实际输出形如 `lr0.001_weight_decay0.1`，注释中是 `lr=0.001_weight_decay=0.1`，可后续统一。
