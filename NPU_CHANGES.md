# NPU 适配改动说明 (Record 4 — Muon Improvements)

基于 modded-nanogpt 的 Record 7 NPU 适配经验，对 Record 4 (commit b356a1fb) 进行 NPU 适配。

Record 4 对应 README 中 "22.3 minutes | Muon improvements | 10/11/24"，由 @kellerjordan0 和 @bozavlado 贡献。

## Record 4 vs Record 7 关键区别

| 特性 | Record 4 | Record 7 |
|---|---|---|
| QKV 投影 | 合并 `c_attn` (3*n_embd) | 分离 Q/K/V |
| 激活函数 | GELU | ReLU² |
| Norm | 手写 rmsnorm | F.rms_norm |
| vocab_size | 50257 | 50304 |
| n_head | 12 (head_dim=64) | 6 (head_dim=128) |
| Attn scaling | `attn_scale = 1/sqrt(2*n_layer)` | 无 |
| 零初始化 | 无 | c_proj 零初始化 |
| QK norm | 无 | 有 |
| Muon | 非分布式 | 分布式 (rank/world_size) |
| num_iterations | 6200 | 5100 |
| warmdown_iters | 1800 | 1450 |

## 一、纯 CUDA→NPU 文本替换（机械性）

| 原始 (CUDA) | 替换为 (NPU) | 位置 |
|---|---|---|
| `import torch._inductor.config as config` | `import torch_npu` | import 区 |
| `x.cuda(), y.cuda()` | `x.npu(), y.npu()` | DataLoader.next_batch() |
| `assert torch.cuda.is_available()` | `assert torch.npu.is_available()` | DDP 初始化 |
| `backend='nccl'` | `backend='hccl'` | init_process_group |
| `f'cuda:{ddp_local_rank}'` | `f'npu:{ddp_local_rank}'` | device 字符串 |
| `torch.cuda.set_device(device)` | `torch.npu.set_device(device)` | 设备设置 |
| `device_type='cuda'` | `device_type='npu'` | torch.amp.autocast |
| `model.cuda()` | `model.npu()` | 模型上设备 |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` | 同步 (共6处) |
| `torch.cuda.max_memory_allocated()` | `torch.npu.max_memory_allocated()` | 显存统计 |
| `nvidia-smi` | `npu-smi info` | 日志记录 |

## 二、非等价改动（不是简单替换）

### 2.1 跳过 torch.compile 和 coordinate_descent_tuning
```python
# 原始
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True
model = torch.compile(model)

# NPU 版：直接删除
# torch.compile 在 NPU 上不完全支持，跳过
```
**影响**：不影响数学正确性，但无算子融合优化，显存占用增大。

### 2.2 移除 @torch.compile 装饰器
```python
# 原始
@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):

# NPU 版：移除装饰器
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
```
**影响**：不影响数学正确性，Muon 的 Newton-Schulz 迭代改为 eager 模式执行。

### 2.3 移除 AdamW fused=True
```python
# 原始
optimizer1 = torch.optim.AdamW(..., fused=True)

# NPU 版
optimizer1 = torch.optim.AdamW(...)
```
**影响**：fused AdamW 是 CUDA 专用优化，NPU 上不支持。数学上等价。

### 2.4 device_batch_size 64→32
```python
# 原始
device_batch_size : int = 64

# NPU 版
device_batch_size : int = 32
```
**配置**：16 卡 NPU 下 train_accumulation_steps = 512 // (32 * 16) = 1，无需梯度累积，总 batch_size 不变。

### 2.5 Rotary Embedding 改为 self 属性
```python
# 原始：register_buffer
self.register_buffer("inv_freq", inv_freq)

# NPU 版：直接 self 属性，避免 buffer 设备同步问题
self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
```
同时对 cos/sin cache 加了 `.bfloat16()` 转换，与 record 7 保持一致。

### 2.6 apply_rotary_emb 添加 type_as
```python
# NPU 版在返回时加上 .type_as(x) 确保类型一致
return torch.cat([y1, y2], 3).type_as(x)
```

### 2.7 验证时使用 autocast context
```python
# 原始：验证时不使用 ctx
with torch.no_grad():
    _, loss = model(x_val, y_val, return_logits=False)

# NPU 版：使用 ctx (与 record 7 一致)
with ctx:
    _, loss = model(x_val, y_val, return_logits=False)
```
**影响**：验证使用 bfloat16 autocast，可能导致 val_loss 微小差异，但与 record 7 行为一致。

### 2.8 run.sh 改动
- `--nproc_per_node=8` → `--nproc_per_node=16`（适配 16 卡 NPU 环境，8张910C，每张2个进程）

## 三、未改动部分（Record 4 特有，保持原样）

- **Muon 非分布式**：Record 4 的 Muon 没有分布式优化（每个 rank 独立计算所有参数的 Newton-Schulz），与 record 7 不同。这保持了 record 4 的原始训练动态。
- **合并 QKV (c_attn)**：保留原始的合并 QKV 投影和分组 Newton-Schulz 处理。
- **GELU 激活**：保留原始的 GELU（record 7 改为 ReLU²）。
- **rmsnorm 手写实现**：保留原始的手写 rmsnorm 函数。
- **attn_scale**：保留原始的 `(1 / (2 * n_layer)**0.5)` 注意力缩放。
- **vocab_size=50257, n_head=12**：保留原始配置。
- **num_iterations=6200, warmdown_iters=1800**：保留原始训练迭代数。

## 四、总结

- 纯替换部分：cuda→npu、nccl→hccl、nvidia-smi→npu-smi
- 非等价部分：跳过 torch.compile（含 @torch.compile）、移除 fused AdamW、device_batch_size 64→32、Rotary 小调整
- 总 batch_size (512 序列 = 524288 tokens) 保持不变
- 16 卡 NPU × B=32 = 512 序列/步，无需梯度累积
- 训练动态完全保留 record 4 原始设计
