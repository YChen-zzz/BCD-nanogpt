"""
Muon 优化器模块
===============
Muon: MomentUm Orthogonalized by Newton-Schulz

参数分配策略:
- 2D 且非 embedding/head 参数: 使用 Muon
- embedding/head 以及非 2D 参数: 使用 AdamW
"""

import torch


# =============================================================================
# 超参数定义 (用于命令行注册和 BCD 搜索)
# =============================================================================

HYPERPARAMS = {
    # Muon 自身超参数 (作用于 transformer.h)
    'lr': {
        'type': float, 'default': 0.001,
        'help': 'adamw 学习率 (transformer.h)',
    },
    'muon_momentum': {
        'type': float, 'default': 0.95,
        'help': 'Muon 动量系数',
    },
    'muon_eps': {
        'type': float, 'default': 1e-7,
        'help': 'Muon Newton-Schulz normalization epsilon',
    },
    'adamw_beta1': {
        'type': float, 'default': 0.9,
        'help': 'Embedding AdamW beta1',
    },
    'adamw_beta2': {
        'type': float, 'default': 0.95,
        'help': 'Embedding AdamW beta2',
    },
    'adamw_eps': {
        'type': float, 'default': 1e-8,
        'help': 'Embedding AdamW epsilon',
    },
    'weight_decay': {
        'type': float, 'default': 0.1,
        'help': 'Embedding AdamW and muon weight decay',
    },
    'muon_lr_adjust': {
        'type': str, 'default': "kimi",
        'help': 'use which lr adjust in muon',
    },
}


# =============================================================================
# Newton-Schulz 正交化后端
# =============================================================================

def zeropower_via_svd(G, steps=None):
    """通过 SVD 分解计算矩阵的零次幂（精确正交化，较慢）"""
    U, S, V = G.svd()
    return U @ V.T


def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    通过 Newton-Schulz 五次迭代计算矩阵正交化
    系数选择为最大化零点斜率，在 bf16 下稳定运行
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    # Newton-Schulz 迭代在 bf16 下进行
    X = G.bfloat16()
    X /= (X.norm() + eps)  # 确保最大奇异值 <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)


# =============================================================================
# Muon Optimizer 类
# =============================================================================

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-Schulz

    内部运行标准 SGD-momentum，然后对每个 2D 参数的更新执行正交化后处理。
    参数:
        lr: 学习率
        momentum: 动量系数
        nesterov: 是否使用 Nesterov 动量
        backend: 正交化方法 ('newtonschulz5' 推荐)
        backend_steps: 迭代步数
        muon_lr_adjust: lr 缩放策略 ('kimi' | '10_mup' | '10_keller_jordan')
    """

    _LR_ADJUST_FNS = {
        "kimi":             lambda s0, s1: 0.2 * max(s0, s1) ** 0.5,
        "10_mup":           lambda s0, s1: 10  * (s0 / s1)   ** 0.5,
        "10_keller_jordan": lambda s0, s1: 10  * max(1, s0 / s1) ** 0.5,
    }

    def __init__(self, params, lr=3e-4, wd=0.1, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5, eps=1e-7, muon_lr_adjust="kimi"):
        if muon_lr_adjust not in self._LR_ADJUST_FNS:
            raise ValueError(f"muon_lr_adjust 不支持 '{muon_lr_adjust}'，可选: {list(self._LR_ADJUST_FNS)}")
        defaults = dict(lr=lr, wd=wd, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps, eps=eps)
        super().__init__(params, defaults)
        self._lr_adjust_fn = self._LR_ADJUST_FNS[muon_lr_adjust]

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['wd']
            zeropower_backend = zeropower_backends[group['backend']]

            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                g = zeropower_backend(g, steps=group['backend_steps'], eps=group['eps'])
                g *= self._lr_adjust_fn(g.size(0), g.size(1))

                p.data.mul_(1 - lr * wd)
                p.data.add_(g, alpha=-lr)


# =============================================================================
# configure 接口
# =============================================================================

def configure(raw_model, args, rank=0, world_size=1):
    """
    创建 Muon 优化器组合

    参数分配:
    - 2D 且非 embedding/head 参数 → Muon
    - embedding/head 以及非 2D 参数 → AdamW

    返回: 优化器列表 [embed_optimizer, muon_optimizer]
    """
    muon_params = []
    adamw_params = []
    for name, p in raw_model.named_parameters():
        if p.ndim >= 2 and 'transformer.wte' not in name and 'lm_head' not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    # Embedding/lm_head 以及非 2D 参数使用 AdamW
    embed_optimizer = torch.optim.AdamW(
        adamw_params,
        lr=args.lr,
        betas=(args.adamw_beta1, args.adamw_beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )

    # Transformer blocks 使用 Muon
    muon_optimizer = Muon(
        muon_params,
        lr=args.lr,
        wd=args.weight_decay,
        eps=args.muon_eps,
        momentum=args.muon_momentum,
        nesterov=True,
        backend='newtonschulz5',
        backend_steps=5,
        muon_lr_adjust=args.muon_lr_adjust,
    )

    return [embed_optimizer, muon_optimizer]
