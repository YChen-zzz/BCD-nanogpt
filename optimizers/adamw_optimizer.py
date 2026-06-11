"""
AdamW 优化器模块
================
所有参数统一使用 AdamW 优化器。

参数分配策略:
- 所有参数使用同一个 AdamW 实例
"""

import torch


# =============================================================================
# 超参数定义 (用于命令行注册和 BCD 搜索)
# =============================================================================

HYPERPARAMS = {
    'lr': {
        'type': float, 'default': 1e-3,
        'help': 'AdamW 学习率',
    },
    'beta1': {
        'type': float, 'default': 0.9,
        'help': 'AdamW beta1',
    },
    'beta2': {
        'type': float, 'default': 0.999,
        'help': 'AdamW beta2',
    },
    'eps': {
        'type': float, 'default': 1e-8,
        'help': 'AdamW epsilon',
    },
    'weight_decay': {
        'type': float, 'default': 0.0,
        'help': 'AdamW weight decay',
    },
}


# =============================================================================
# configure 接口
# =============================================================================

def configure(raw_model, args, rank=0, world_size=1):
    """
    创建 AdamW 优化器

    参数分配:
    - 所有参数统一使用 AdamW

    返回: 优化器列表 [adamw_optimizer]
    """
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    return [optimizer]
