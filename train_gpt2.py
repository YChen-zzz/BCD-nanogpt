"""
NanoGPT 预训练脚本 (NPU 版本) — nano-compile fast path
==============================
- 模型: 通过 --model 指定 (如 124m)，自动映射到对应配置
- 优化器: 通过 --optimizer 指定 (如 muon, adamw)，每个 optimizer 自带参数分配逻辑
- QKV 分开为 c_q, c_k, c_v
- 固定 random seed 以保证可复现性
- activation 使用 bf16, 权重/optimizer state/梯度 使用 fp32

用法:
    torchrun --standalone --nproc_per_node=16 train_gpt2.py --optimizer muon [超参数...]
    torchrun --standalone --nproc_per_node=16 train_gpt2.py --optimizer adamw --lr 1e-3
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()  # 尽早读取本文件代码，用于日志记录
import uuid
import glob
import time
import json
import inspect
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch_npu
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from optimizers import load_optimizer_module


# =============================================================================
# 工具函数
# =============================================================================

def set_seed(seed):
    """固定所有随机种子，确保训练可复现"""
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# GPT-2 模型定义
# =============================================================================

class Rotary(nn.Module):
    """旋转位置编码 (RoPE) — 静态预计算, 消除 forward 动态分支"""

    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_buf", freqs.cos().bfloat16()[None, :, None, :])
        self.register_buffer("sin_buf", freqs.sin().bfloat16()[None, :, None, :])
        cos_full = torch.cat([freqs.cos(), freqs.cos()], dim=-1).bfloat16()
        neg_sin_full = -torch.cat([freqs.sin(), freqs.sin()], dim=-1).bfloat16()
        self.register_buffer("cos_full_buf", cos_full[None, :, None, :])
        self.register_buffer("neg_sin_full_buf", neg_sin_full[None, :, None, :])
        self.use_npu_rotary = False

    def forward(self, x):
        seq_len = x.shape[1]
        return self.cos_buf[:, :seq_len], self.sin_buf[:, :seq_len]

    def apply(self, x):
        seq_len = x.shape[1]
        if self.use_npu_rotary and hasattr(torch_npu, 'npu_rotary_mul') and x.device.type == 'npu':
            cos_full = self.cos_full_buf[:, :seq_len]
            neg_sin_full = self.neg_sin_full_buf[:, :seq_len]
            return torch_npu.npu_rotary_mul(x, cos_full, neg_sin_full).type_as(x)
        cos, sin = self.cos_buf[:, :seq_len], self.sin_buf[:, :seq_len]
        return apply_rotary_emb(x, cos, sin)


def apply_rotary_emb(x, cos, sin):
    """应用旋转位置编码"""
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def rmsnorm(x0, eps=1e-6):
    """RMS 归一化，内部使用 fp32 计算以保证精度"""
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class FastRMSNorm(nn.Module):
    """无参数 RMSNorm；可切到 NPU fused RMSNorm kernel"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.use_npu_rms_norm = False
        self.register_buffer("weight", torch.ones(dim), persistent=False)

    def forward(self, x):
        if self.use_npu_rms_norm and hasattr(torch_npu, 'npu_rms_norm') and x.device.type == 'npu':
            return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0].type_as(x)
        return rmsnorm(x, self.eps)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力模块
    支持两种模式:
    - 分离 QKV: c_q, c_k, c_v (方便不同 optimizer 统一处理)
    - 合并 QKV: c_qkv (减少 kernel launch, 提升计算密度)
    """

    def __init__(self, config, fused_qkv=False):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.fused_qkv = fused_qkv
        if fused_qkv:
            self.c_qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        else:
            self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim, max_seq_len=2048)
        self.attn_scale = self.head_dim ** -0.5
        self.use_npu_attention = False
        self.causal_mask_cached = None
        self.causal_mask_seq_len = None

    def _get_causal_mask(self, seq_len, device):
        if self.causal_mask_cached is None or self.causal_mask_seq_len != seq_len:
            self.causal_mask_seq_len = seq_len
            self.causal_mask_cached = torch.ones(
                (seq_len, seq_len), device=device, dtype=torch.bool
            ).triu(1)
        return self.causal_mask_cached

    def forward(self, x):
        B, T, C = x.size()
        if self.fused_qkv:
            qkv = self.c_qkv(x).view(B, T, 3, self.n_head, self.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        q = self.rotary.apply(q)
        k = self.rotary.apply(k)

        if self.use_npu_attention and hasattr(torch_npu, 'npu_fusion_attention') and q.device.type == 'npu':
            y = torch_npu.npu_fusion_attention(
                q, k, v,
                head_num=self.n_head,
                input_layout="BSND",
                atten_mask=self._get_causal_mask(T, q.device),
                scale=self.attn_scale,
                keep_prob=1.0,
                pre_tockens=T,
                next_tockens=0,
                sparse_mode=0,
            )[0]
            y = y.contiguous().view_as(x)
        else:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
            )
            y = y.transpose(1, 2).contiguous().view_as(x)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """前馈网络，使用 GELU 激活函数（对齐 record4 原始设计）"""

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer Block: RMSNorm + Attention + RMSNorm + MLP"""

    def __init__(self, config, fused_qkv=False):
        super().__init__()
        self.attn = CausalSelfAttention(config, fused_qkv=fused_qkv)
        self.mlp = MLP(config)
        self.attn_norm = FastRMSNorm(config.n_embd)
        self.mlp_norm = FastRMSNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


# =============================================================================
# 模型配置 (通过 --model 指定)
# =============================================================================

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 10
    n_embd: int = 640


# 模型大小到配置的映射表
# 使用时通过 --model 124m 来选择
MODEL_CONFIGS = {
    '124m': GPTConfig(vocab_size=50257, n_layer=12, n_head=10, n_embd=640),
    '300m': GPTConfig(vocab_size=50257, n_layer=16, n_head=16, n_embd=1024),
    '480m': GPTConfig(vocab_size=50257, n_layer=18, n_head=16, n_embd=1280),
}


class GPT(nn.Module):
    """
    GPT-2 模型
    - 无 weight tying: wte 和 lm_head 参数独立
    - optimizer 模块自行决定参数分配
    """

    def __init__(self, config, fused_qkv=False):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config, fused_qkv=fused_qkv) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.final_norm = FastRMSNorm(config.n_embd)

    def enable_npu_fused_kernels(self, rmsnorm=True, rope=True, attention=False):
        """启用 NPU 融合算子"""
        for module in self.modules():
            if isinstance(module, FastRMSNorm):
                module.use_npu_rms_norm = rmsnorm
            elif isinstance(module, Rotary):
                module.use_npu_rotary = rope
            elif isinstance(module, CausalSelfAttention):
                module.use_npu_attention = attention

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)

        for block in self.transformer.h:
            x = block(x)
        x = self.final_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.float()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            logits = logits.float()
            loss = None

        if not return_logits:
            logits = None
        return logits, loss


# =============================================================================
# 分布式数据加载器
# =============================================================================

def _peek_data_shard(filename):
    """读取数据分片 header，返回 token 数量"""
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: data .bin magic number mismatch!")
        exit(1)
    assert header[1] == 1, "unsupported version"
    return header[2]


def _load_data_shard(filename):
    """加载完整数据分片"""
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "token 数量与 header 不匹配"
    return tokens


class DistributedDataLoader:
    """
    分布式数据加载器
    按文件顺序加载分片，每个进程读取不同偏移位置
    固定 seed 后所有 run 数据顺序完全一致
    """

    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"未找到匹配文件: {filename_pattern}"

        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.to("npu", non_blocking=True), y.to("npu", non_blocking=True)


# =============================================================================
# 命令行参数解析 (两阶段: 先解析 --optimizer，再注册对应超参数)
# =============================================================================

def parse_args():
    """
    两阶段参数解析:
    1. 先解析 --optimizer 和 --model 确定使用哪个优化器和模型
    2. 再注册该优化器特有的超参数
    """
    # 第一阶段：解析基础参数
    parser = argparse.ArgumentParser(description='NanoGPT 预训练 (NPU)')

    # --- 模型选择 ---
    parser.add_argument('--model', type=str, default='124m',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='模型大小')

    # --- 优化器选择 ---
    parser.add_argument('--optimizer', type=str, default='muon',
                        help='优化器名称 (muon, adamw, ...)')

    # --- 数据相关 ---
    parser.add_argument('--input_bin', type=str, default='/models/share/chenyupeng/chenyupeng/nanogpt_shipei/cached_fineweb100B/fineweb_train_*.bin',
                        help='训练数据路径 (glob 模式)')
    parser.add_argument('--input_val_bin', type=str, default='/models/share/chenyupeng/chenyupeng/nanogpt_shipei/cached_fineweb100B/fineweb_val_*.bin',
                        help='验证数据路径 (glob 模式)')

    # --- 训练规模 ---
    parser.add_argument('--batch_size', type=int, default=8*64,
                        help='全局 batch size (序列数)')
    parser.add_argument('--device_batch_size', type=int, default=None,
                        help='每设备 batch size (默认 batch_size/world_size，使梯度累积为 1)')
    parser.add_argument('--sequence_length', type=int, default=1024,
                        help='序列长度')
    parser.add_argument('--chinchilla_multiplier', type=float, default=1.0,
                        help='Chinchilla 倍数 (>0 时自动计算 num_iterations)')
    parser.add_argument('--num_iterations', type=int, default=4704,
                        help='训练步数 (chinchilla_multiplier>0 时自动覆盖)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度 norm clip 阈值 (<=0 表示不裁剪)')

    # --- LR Scheduler ---
    parser.add_argument('--warmup_fraction', type=float, default=0.05,
                        help='学习率 warmup 阶段占总步数的比例 '
                             '(warmup_iters = warmup_fraction * num_iterations)')
    parser.add_argument('--warmup_iters', type=int, default=None,
                        help='学习率 warmup 步数 (兼容旧参数；指定后覆盖 warmup_fraction)')
    parser.add_argument('--wsd_fraction', type=float, default=0.2,
                        help='WSD scheduler 中 decay 阶段占总步数的比例 '
                             '(warmdown_iters = wsd_fraction * num_iterations)')

    # --- 评估与日志 ---
    parser.add_argument('--val_loss_every', type=int, default=125,
                        help='每多少步评估验证 loss (0=仅最后)')
    parser.add_argument('--val_tokens', type=int, default=10485760,
                        help='验证集 token 数量')
    parser.add_argument('--save_every', type=int, default=0,
                        help='每多少步保存 checkpoint (0=仅最后)')
    parser.add_argument('--output_dir', type=str, default='',
                        help='日志和 checkpoint 输出目录 (默认自动生成 logs/{uuid})')

    # --- 可复现性 ---
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # --- NPU 融合算子 ---
    parser.add_argument('--fused_rmsnorm', type=int, default=0, choices=[0, 1],
                        help='启用 NPU fused RMSNorm (torch_npu.npu_rms_norm)')
    parser.add_argument('--fused_rope', type=int, default=0, choices=[0, 1],
                        help='启用 NPU fused Rotary (torch_npu.npu_rotary_mul)')
    parser.add_argument('--fused_attention', type=int, default=0, choices=[0, 1],
                        help='启用 NPU fusion attention (torch_npu.npu_fusion_attention)')
    parser.add_argument('--fused_qkv', type=int, default=0, choices=[0, 1],
                        help='合并 QKV 为单个 Linear (减少 kernel launch)')
    parser.add_argument('--ddp_gradient_as_bucket_view', type=int, default=0, choices=[0, 1],
                        help='DDP gradient_as_bucket_view 优化')
    parser.add_argument('--ddp_static_graph', type=int, default=0, choices=[0, 1],
                        help='DDP static_graph 优化')


    # 第二阶段：解析 optimizer 名称后，注册其超参数
    # 使用 parse_known_args 先拿到 optimizer 名称
    pre_args, remaining = parser.parse_known_args()

    # 注册 optimizer 特有的超参数
    opt_module = load_optimizer_module(pre_args.optimizer)
    for param_name, param_info in opt_module.HYPERPARAMS.items():
        parser.add_argument(
            f'--{param_name}',
            type=param_info['type'],
            default=param_info['default'],
            help=param_info.get('help', ''),
        )

    return parser.parse_args()


# =============================================================================
# 主训练流程
# =============================================================================

def main():
    args = parse_args()

    # ---- 初始化分布式环境 ----
    assert torch.npu.is_available(), "需要 NPU 环境"
    dist.init_process_group(backend='hccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'npu:{ddp_local_rank}'
    torch.npu.set_device(device)
    master_process = (ddp_rank == 0)
    if master_process:
        print(f"使用设备: {device}, 总进程数: {ddp_world_size}")

    # ---- 固定随机种子 ----
    set_seed(args.seed)

    # ---- 初始化模型 ----
    model_config = MODEL_CONFIGS[args.model]
    model = GPT(model_config, fused_qkv=bool(args.fused_qkv))
    model = model.npu()

    # 启用 NPU 融合算子
    if args.fused_rmsnorm or args.fused_rope or args.fused_attention:
        model.enable_npu_fused_kernels(
            rmsnorm=bool(args.fused_rmsnorm),
            rope=bool(args.fused_rope),
            attention=bool(args.fused_attention),
        )
        if master_process:
            print(f"NPU fused ops: rmsnorm={args.fused_rmsnorm}, "
                  f"rope={args.fused_rope}, attention={args.fused_attention}, "
                  f"fused_qkv={args.fused_qkv}")

    # DDP 包装
    ddp_kwargs = dict(device_ids=[ddp_local_rank])
    if args.ddp_gradient_as_bucket_view:
        ddp_kwargs['gradient_as_bucket_view'] = True
    if args.ddp_static_graph:
        ddp_kwargs['static_graph'] = True
    model = DDP(model, **ddp_kwargs)
    raw_model = model.module
    total_params = sum(p.numel() for p in raw_model.parameters())
    ctx = torch.amp.autocast(device_type='npu', dtype=torch.bfloat16)

    if master_process:
        print(f"模型: {args.model}, 参数量: {total_params/1e6:.2f}M")
        print(f"配置: {asdict(model_config)}")

    # ---- 计算训练步数 ----
    if args.device_batch_size is None:
        assert args.batch_size % ddp_world_size == 0, \
            "batch_size 必须能被 world_size 整除，才能默认使用梯度累积 1"
        args.device_batch_size = args.batch_size // ddp_world_size

    B, T = args.device_batch_size, args.sequence_length
    assert B > 0, "device_batch_size 必须大于 0"
    sequences_per_micro_step = B * ddp_world_size
    assert args.batch_size % sequences_per_micro_step == 0, \
        (f"batch_size={args.batch_size} 必须能被 "
         f"device_batch_size*world_size={B}*{ddp_world_size}={sequences_per_micro_step} 整除")
    train_accumulation_steps = args.batch_size // sequences_per_micro_step
    tokens_per_step = args.batch_size * T

    # Chinchilla 倍数自动计算
    if args.chinchilla_multiplier > 0:
        total_tokens = int(args.chinchilla_multiplier * total_params * 20)
        args.num_iterations = total_tokens // tokens_per_step
        if master_process:
            print(f"Chinchilla {args.chinchilla_multiplier}x: "
                  f"total_tokens={total_tokens/1e9:.2f}B, steps={args.num_iterations}")

    if args.warmup_iters is None:
        warmup_iters = int(args.warmup_fraction * args.num_iterations)
    else:
        warmup_iters = args.warmup_iters
        args.warmup_fraction = warmup_iters / args.num_iterations if args.num_iterations > 0 else 0.0

    # warmdown_iters = wsd_fraction * (总步数 - warmup步数)，确保 warmdown 不与 warmup 重叠
    warmdown_iters = int(args.wsd_fraction * (args.num_iterations - warmup_iters))
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)

    if master_process:
        print(f"训练: steps={args.num_iterations}, tokens/step={tokens_per_step}, "
              f"total={args.num_iterations * tokens_per_step / 1e9:.2f}B tokens")
        print(f"Batch: global={args.batch_size}, per_device={B}, "
              f"accumulation={train_accumulation_steps}")
        print(f"Scheduler: warmup={warmup_iters}, warmdown={warmdown_iters}")
        print(f"Grad clip: {args.grad_clip}")
        print(f"Optimizer: {args.optimizer}, seed={args.seed}")

    # ---- 加载数据 ----
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    if master_process:
        print(f"训练数据: {train_loader.ntok_total/1e9:.2f}B tokens, "
              f"{len(train_loader.files)} 个文件")
    x, y = train_loader.next_batch()

    # ---- 初始化优化器 (由 optimizer 模块自行决定参数分配) ----
    opt_module = load_optimizer_module(args.optimizer)
    optimizers = opt_module.configure(raw_model, args, rank=ddp_rank, world_size=ddp_world_size)

    # ---- LR Scheduler: 线性 warmup + 恒定 + 线性 warmdown ----
    def get_lr(it):
        assert it <= args.num_iterations
        if warmup_iters > 0 and it < warmup_iters:
            return (it + 1) / warmup_iters
        warmdown_start = args.num_iterations - warmdown_iters
        if it < warmdown_start:
            return 1.0
        elif warmdown_iters > 0:
            return (args.num_iterations - it) / warmdown_iters
        else:
            return 1.0

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    # ---- 日志初始化 ----
    if master_process:
        run_id = str(uuid.uuid4())
        # 如果指定了 output_dir，使用它；否则自动生成
        if args.output_dir:
            logdir = args.output_dir
        else:
            logdir = f'logs/{run_id}'
        os.makedirs(logdir, exist_ok=True)
        logfile = os.path.join(logdir, 'train.log')
        # 保存完整配置到 JSON（BCD 搜索引擎使用）
        config_dict = vars(args).copy()
        config_dict['warmup_iters'] = warmup_iters
        config_dict['warmdown_iters'] = warmdown_iters
        config_dict['train_accumulation_steps'] = train_accumulation_steps
        config_dict['total_params'] = total_params
        config_dict['run_id'] = run_id
        config_dict['model_config'] = asdict(model_config)
        with open(os.path.join(logdir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        with open(logfile, "w") as f:
            f.write('='*100 + '\n')
            f.write(code)
            f.write('='*100 + '\n')
            f.write(f"PyTorch {torch.version.__version__}\n")
            f.write(f"Config: {json.dumps(config_dict, indent=2)}\n")
            f.write('='*100 + '\n')

    # ---- 预缓存参数列表 ----
    all_params = list(model.parameters())
    grad_params = [p for p in all_params if p.requires_grad]

    # ---- 训练循环 ----
    training_time_ms = 0
    torch.npu.synchronize()
    t0 = time.time()
    train_loader.reset()

    for step in range(args.num_iterations + 1):
        last_step = (step == args.num_iterations)
        # 前 10 步预热，不计入计时
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1

        # ---- 验证集评估 ----
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.npu.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            for _ in range(val_steps):
                x_val, y_val = val_loader.next_batch()
                with ctx:
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
                    del loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            if master_process:
                current_lrs = {f'opt{i}': sched.get_last_lr()[0] for i, sched in enumerate(schedulers)}
                lr_str = ' '.join(f'lr_{k}:{v:.6f}' for k, v in current_lrs.items())
                train_time_min = training_time_ms / 60000
                step_avg_ms = training_time_ms / (timed_steps - 1)
                predict_time_min = step_avg_ms * args.num_iterations / 60000
                msg = (f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} '
                       f'{lr_str} '
                       f'train_time:{train_time_min:.2f}/{predict_time_min:.2f}min '
                       f'step_avg:{step_avg_ms:.2f}ms')
                print(msg)
                with open(logfile, "a") as f:
                    f.write(msg + '\n')
            torch.npu.synchronize()
            t0 = time.time()

        # ---- 保存 checkpoint ----
        # if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        #     torch.npu.synchronize()
        #     training_time_ms += 1000 * (time.time() - t0)
        #     # ckpt = dict(step=step, code=code,
        #     #             model=raw_model.state_dict(),
        #     #             optimizers=[opt.state_dict() for opt in optimizers])
        #     ckpt = dict(step=step, code=code,
        #                 model=raw_model.state_dict())
        #     #            optimizers=[opt.state_dict() for opt in optimizers])
        #     torch.save(ckpt, os.path.join(logdir, f'state_step{step:06d}.pt'))
        #     torch.npu.synchronize()
        #     t0 = time.time()

        if last_step:
            break

        # ---- 训练步 ----
        model.train()
        for i in range(1, train_accumulation_steps + 1):
            with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss = loss.detach()
            x, y = train_loader.next_batch()
            if i < train_accumulation_steps:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
        # 梯度平均
        for p in grad_params:
            if p.grad is not None:
                p.grad /= train_accumulation_steps
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
        # 更新参数
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        model.zero_grad(set_to_none=True)

        # ---- 打印训练 loss ----
        if master_process:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            current_lrs = {f'opt{i}': sched.get_last_lr()[0] for i, sched in enumerate(schedulers)}
            lr_str = ' '.join(f'lr_{k}:{v:.6f}' for k, v in current_lrs.items())
            approx_time_min = approx_time / 60000
            step_avg_ms = approx_time / timed_steps
            predict_time_min = step_avg_ms * args.num_iterations / 60000
            msg = (f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} "
                   f"{lr_str} "
                   f"train_time:{approx_time_min:.2f}/{predict_time_min:.2f}min "
                   f"step_avg:{step_avg_ms:.2f}ms")
            print(msg)
            with open(logfile, "a") as f:
                f.write(msg + '\n')

    # ---- 训练结束，保存结果 JSON ----
    if master_process:
        print(f"峰值显存: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB")
        result = {
            'run_id': run_id,
            'final_val_loss': val_loss.item(),
            'total_steps': args.num_iterations,
            'config': vars(args),
        }
        result_path = os.path.join(logdir, 'result.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存: {result_path}")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
