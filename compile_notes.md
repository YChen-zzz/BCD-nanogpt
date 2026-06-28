Step 3: torch.compile 与 NPU 对比分析

torch.compile 工作流程 (GPU)

torch.compile 是 PyTorch 2.0 引入的端到端编译优化，分三步：

1. TorchDynamo (前端): 捕获 Python bytecode，生成 torch.fx.Graph (FX IR)
2. AOTAutograd: 在 FX Graph 上做自动微分，生成前向 + 反向图
3. Inductor (后端): 把 FX Graph 编译成可执行 kernel
   - GPU -> 生成 Triton kernel
   - CPU -> 生成 C++/OpenMP 代码

效果示例: rmsnorm 中的 .float() -> .pow(2) -> .mean() -> rsqrt() -> * -> .type_as()
六个小算子会被 Inductor 自动融合为一个 Triton kernel，减少 kernel launch 和显存读写。


NPU 为什么用不了 torch.compile

Inductor 后端只能生成 Triton (NVIDIA GPU) 或 C++ (CPU) 代码，
无法为昇腾 DaVinci 架构生成指令，因此 torch.compile 在 NPU 上不可用。

华为的替代方案 (torchair) 是替换 Inductor 后端，接入 CANN 的 GE (Graph Engine):

  GPU 路线: Dynamo -> FX Graph -> Inductor -> Triton kernel
  NPU 路线: Dynamo -> FX Graph -> torchair -> GE/CANN

torchair 目前仍为实验性项目，尚未成熟。


torch.fx 在 NPU 上的可用性

torch.fx 是 torch.compile 的底层图 IR 工具，本身与硬件无关:

- torch.fx.symbolic_trace: 符号追踪，NPU 上正常工作
- torch.fx.Graph 的图变换、节点替换: 纯 Python 操作，不涉及硬件
- 不可用的部分: 仅 Inductor 编译后端 (Triton kernel 生成)


NPU 的替代优化路线

由于 torch.compile 不可用，NPU 侧的算子融合需要手动完成:

- 手动调用 CANN 融合算子: 如 npu_fused_rms 替代逐步计算的 rmsnorm
- torch_npu graph mode / jit trace: 替代 Dynamo 的图捕获
- CANN 算子库自身优化: 底层 matmul/attention kernel 由 CANN 管理

本质上是用人工劳动弥补编译器的缺失:
GPU 侧一行 torch.compile(model) 自动完成的融合，NPU 侧需要逐个算子手动替换。


GPU 侧优化项实际生效情况

- torch.set_float32_matmul_precision("high"): 未生效
  所有 matmul 都在 bf16 autocast 下运行，不存在 fp32 matmul，此设置为空操作。
- cudnn.benchmark = True: 生效，cuDNN autotuning 选择最优 kernel (NPU 无 cuDNN，不适用)
- torch.compile(model): 生效，Inductor 自动融合算子 (NPU 不可用)
- F.scaled_dot_product_attention: GPU/NPU 两侧都使用，自动 dispatch 到各自的 FlashAttention 实现
- fused AdamW: 已在 adamw_optimizer.py 中启用 (fused=True)，不影响数值结果，纯性能优化
