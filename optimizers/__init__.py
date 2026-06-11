"""
optimizers 包
=============
每个 optimizer 模块需要实现以下接口：

1. HYPERPARAMS: dict
   定义该 optimizer 所有可调超参数的名称、默认值、类型
   用于命令行参数注册和 BCD 搜索

2. configure(raw_model, args, rank, world_size) -> list[torch.optim.Optimizer]
   根据模型结构和超参数创建优化器列表
   负责决定哪些参数用哪个优化器

使用示例:
    from optimizers import load_optimizer_module
    opt_module = load_optimizer_module('muon')
    optimizers = opt_module.configure(raw_model, args, rank, world_size)
"""

import importlib

# 支持的优化器注册表
OPTIMIZER_REGISTRY = {
    'muon': 'optimizers.muon_optimizer',
    'adamw': 'optimizers.adamw_optimizer',
}


def load_optimizer_module(name):
    """根据名称加载对应的 optimizer 模块"""
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"不支持的优化器: {name}\n"
            f"可用: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    return importlib.import_module(OPTIMIZER_REGISTRY[name])


def register_optimizer_args(parser, name):
    """将指定 optimizer 的超参数注册为命令行参数"""
    module = load_optimizer_module(name)
    for param_name, param_info in module.HYPERPARAMS.items():
        arg_name = f'--{param_name}'
        parser.add_argument(
            arg_name,
            type=param_info['type'],
            default=param_info['default'],
            help=param_info.get('help', ''),
        )
