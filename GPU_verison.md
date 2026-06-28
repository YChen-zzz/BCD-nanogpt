Step 1 Migration Complete

train_gpt2.py Changes

- import torch_npu -> removed
- torch.npu.* -> torch.cuda.* (synchronize, set_device, manual_seed_all, max_memory_allocated, is_available)
- .npu() -> .cuda() (model, tensor transfers)
- DDP backend: hccl -> nccl
- autocast: device_type='npu' -> device_type='cuda'
- Default nproc_per_node: 16 -> 8
- Docstring/description: NPU -> GPU

bcd_search.py Changes

- Default nproc: 16 -> 8
- NPUS_PER_NODE -> GPUS_PER_NODE
- Removed all Ascend toolkit env sourcing (set_env.sh, init_env.sh, nnal/atb)
- Removed conda env activation from launcher scripts (user manages own env on GPU)

optimizers/ -- No changes needed

- muon_optimizer.py and adamw_optimizer.py use pure PyTorch APIs, already GPU-compatible

Backups

- train_gpt2_npu_backup.py
- bcd_search_npu_backup.py

Step 2 TODO (compile & optimization)

- Add torch.compile(model) for kernel fusion
- Consider torch.set_float32_matmul_precision('high') for tf32 matmul
- Explicit FlashAttention config if needed
- Update data paths (--input_bin, --input_val_bin) for GPU server
- Adjust batch_size default for GPU memory



Step 2 完成。以下是改动总结：

  Step 2 Changes in train_gpt2.py

  - Line 382: torch.set_float32_matmul_precision("high") -- 启用 tf32 matmul，在 Ampere+ GPU 上对 fp32 矩阵运算有约 2-3x 加速，精度损
    失可忽略
  - Line 46: cudnn.benchmark = True -- 序列长度固定为 1024，开启 cuDNN autotuning 可以选择最优 kernel
  - Line 400: model = torch.compile(model) -- 在 DDP 包装之前 compile，触发 Triton/inductor 内核融合
  Note on deterministic = True + benchmark = True

  - 目前两者同时为 True，deterministic 会限制 cuDNN 只用确定性算法但 benchmark 仍可在这些算法中选最快的。如果你发现 compile 后首几步编

  需要你在 GPU 上验证的事项：

  - fineweb_val_*.bin 文件是否存在于同一数据目录
  - batch_size=512 对 8 卡 GPU 的显存是否足够（每卡 64 个序列 x 1024 tokens）
  - 首次运行 compile 编译需要 ~1-2 分钟 warmup