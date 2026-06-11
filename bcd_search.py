"""
BCD (Block Coordinate Descent) 超参数搜索引擎
=============================================
对每个超参数逐一搜索最优值，循环直到收敛。

用法:
    python bcd_search.py --config configs/search_muon.yaml
    python bcd_search.py --config configs/search_muon.yaml --dry_run

去重逻辑:
    用完整超参数组合作为 key，跨 round 自动复用相同配置的结果。
    例如 Round 1 跑过 {lr=4e-3, wd=0, eps=1e-8}，Round 2 如果
    又需要同样的配置，直接跳过。

目录结构:
    bcd_results/muon/
    ├── bcd_history.json                    # 全局搜索历史
    ├── round_1_result.json                 # 每轮最优结果
    ├── runs/                               # 每次训练的独立文件夹
    │   ├── muon_lr=0.0001_adamw_lr=0.0036_.../
    │   │   ├── config.json
    │   │   ├── result.json
    │   │   ├── train.log
    │   │   └── state_step004704.pt (可选)
    │   └── ...
    └── final_result.json
"""

import os
import json
import yaml
import time
import subprocess
import argparse
import shlex
from pathlib import Path


# =============================================================================
# 工具函数
# =============================================================================

def params_to_dir_name(params):
    """
    将超参数字典转为文件夹名
    例如: {muon_lr: 0.0001, adamw_lr: 0.0036} -> "muon_lr=0.0001_adamw_lr=0.0036"
    """
    parts = []
    for key in sorted(params.keys()):
        val = params[key]
        # 浮点数去掉末尾多余的零
        if isinstance(val, float):
            val = f"{val:g}"
        parts.append(f"{key}{val}")
    return "_".join(parts)


def params_to_key(params):
    """
    将超参数字典转为可哈希的 key（用于去重）
    对 key 排序后转为 tuple，确保顺序无关
    """
    return tuple(sorted((k, str(v)) for k, v in params.items()))


def write_json_atomic(path, data):
    """先写临时文件，再原子替换目标 JSON，避免留下半截文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def wait_for_json(path, timeout=300, interval=2):
    """等待 JSON 文件出现并可完整读取，用于共享文件系统上的同步。"""
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as exc:
                last_error = exc
        time.sleep(interval)
    raise TimeoutError(f"等待 JSON 文件超时: {path}, last_error={last_error}")


def start_or_join_control_session(output_dir, launcher, is_metadata_writer, dry_run, timeout=300, interval=2):
    """multi 模式下建立本次 BCD 搜索的共享决策目录。"""
    if launcher != 'multi' or dry_run:
        return None

    started_at = time.time()
    control_root = os.path.join(output_dir, 'control')
    session_path = os.path.join(control_root, 'session.json')
    if is_metadata_writer:
        session = {
            'search_id': time.strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}',
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'created_at_unix': time.time(),
        }
        write_json_atomic(session_path, session)
    else:
        deadline = time.time() + timeout
        session = None
        while time.time() < deadline:
            try:
                candidate = wait_for_json(session_path, timeout=interval, interval=interval)
            except TimeoutError:
                continue
            if candidate.get('created_at_unix', 0) >= started_at - 600:
                session = candidate
                break
            time.sleep(interval)
        if session is None:
            raise TimeoutError(f"等待新的 multi control session 超时: {session_path}")

    control_dir = os.path.join(control_root, session['search_id'])
    os.makedirs(control_dir, exist_ok=True)
    return control_dir


def decision_path(control_dir, task_idx):
    return os.path.join(control_dir, f'decision_{task_idx:06d}.json')


def validate_decision(decision, task_idx, round_idx, param_name, value, run_params):
    expected_key = params_to_key(run_params)
    actual_key = params_to_key(decision['all_params'])
    if (
        decision['task_idx'] != task_idx
        or decision['round'] != round_idx
        or decision['param_name'] != param_name
        or str(decision['param_value']) != str(value)
        or actual_key != expected_key
    ):
        raise RuntimeError(
            "multi 决策文件和当前 grid point 不匹配: "
            f"decision={decision}, expected_task={task_idx}, "
            f"expected_round={round_idx}, expected_param={param_name}, "
            f"expected_value={value}, expected_params={run_params}"
        )


# =============================================================================
# 搜索历史管理（断点续搜 + 跨 round 去重）
# =============================================================================

def load_history(output_dir):
    """
    加载搜索历史，构建 {完整超参 key: val_loss} 字典
    用完整超参组合去重，而不是 (round, param_name, value)
    """
    history_file = os.path.join(output_dir, 'bcd_history.json')
    if not os.path.exists(history_file):
        return {}, []

    with open(history_file, 'r') as f:
        history = json.load(f)

    # 用完整超参组合构建去重字典
    completed = {}
    for entry in history:
        key = params_to_key(entry['all_params'])
        completed[key] = entry['val_loss']

    return completed, history


def save_history_entry(output_dir, entry):
    """追加一条搜索记录到 bcd_history.json，按完整超参组合去重"""
    history_file = os.path.join(output_dir, 'bcd_history.json')
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []

    entry_key = params_to_key(entry['all_params'])
    for old_entry in history:
        if params_to_key(old_entry['all_params']) == entry_key:
            print(f"    [history] 已存在相同完整配置，跳过追加: {entry['run_dir']}")
            return False

    history.append(entry)
    write_json_atomic(history_file, history)

    return True


# =============================================================================
# 训练执行
# =============================================================================

def build_python_args(all_args):
    """将参数字典转为 train_gpt2.py 的命令行参数列表"""
    py_args = []
    for key, value in all_args.items():
        py_args.extend([f'--{key}', str(value)])
    return py_args


def build_train_command(base_args, hyperparams, output_dir, nproc=16, launcher='single'):
    """
    构建训练命令
    base_args: 固定参数 (optimizer, model, seed, ...)
    hyperparams: 当前超参数值
    output_dir: 该 run 的输出目录
    launcher:
      - single: 单机 torchrun --standalone --nproc_per_node=nproc
      - multi:  多机 torchrun，依赖 /models/share/init_env.sh 提供的环境变量
    """
    repo_dir = str(Path(__file__).resolve().parent)
    all_args = {**base_args, **hyperparams, 'output_dir': output_dir}
    train_args = shlex.join(['train_gpt2.py'] + build_python_args(all_args))

    if launcher == 'single':
        torchrun_cmd = (
            f'torchrun --standalone --nproc_per_node={nproc} {train_args}'
        )
        script = "\n".join([
            f"cd {shlex.quote(repo_dir)}",
            "source /root/miniconda3/etc/profile.d/conda.sh",
            "conda activate llm_test",
            "source /usr/local/Ascend/ascend-toolkit/set_env.sh",
            torchrun_cmd,
        ])
    elif launcher == 'multi':
        torchrun_cmd = (
            "torchrun "
            "--nproc_per_node ${NPUS_PER_NODE} "
            "--nnodes ${TOTAL_NODES} "
            "--node_rank ${CURRENT_NODE_RANK} "
            "--master_addr ${MASTER_IP} "
            "--master_port ${MASTER_PORT} "
            f"{train_args}"
        )
        script = "\n".join([
            f"cd {shlex.quote(repo_dir)}",
            "source /root/miniconda3/etc/profile.d/conda.sh",
            "conda activate llm_test",
            "source /models/share/init_env.sh",
            "source /usr/local/Ascend/ascend-toolkit/set_env.sh",
            "source /usr/local/Ascend/nnal/atb/set_env.sh",
            torchrun_cmd,
        ])
    else:
        raise ValueError(f"未知 launcher: {launcher}，可选: single, multi")

    return ['bash', '-lc', script]


def run_training(cmd, run_dir, dry_run=False, result_timeout=300, result_interval=2):
    """
    执行一次训练，返回 val_loss（或 None 表示失败）
    训练结果存储在 run_dir/result.json 中
    """
    print(f"\n{'='*80}")
    print(f"执行训练: {shlex.join(cmd)}")
    print(f"输出目录: {run_dir}")
    print(f"{'='*80}\n")

    if dry_run:
        print("[DRY RUN] 跳过实际训练")
        return None

    ret = subprocess.run(cmd, capture_output=False, text=True)
    if ret.returncode != 0:
        print(f"[ERROR] 训练失败，返回码: {ret.returncode}")
        return None

    # 读取结果。multi-node 训练里 result.json 可能由 rank0 写入，共享文件系统会有轻微延迟。
    result_path = os.path.join(run_dir, 'result.json')
    try:
        result = wait_for_json(result_path, timeout=result_timeout, interval=result_interval)
    except TimeoutError as exc:
        print(f"[ERROR] 未找到结果文件: {result_path} ({exc})")
        return None
    return result['final_val_loss']


# =============================================================================
# BCD 搜索主逻辑
# =============================================================================

def get_metadata_writer(config, launcher):
    """
    多机模式下每个节点都需要启动 torchrun，但只有一个节点写共享搜索元数据。
    这样可以避免多个 bcd_search.py 进程向同一个 bcd_history.json 追加重复记录。
    """
    if launcher != 'multi':
        return True, None

    rank_name = config.get('node_rank_env', 'CURRENT_NODE_RANK')
    rank_value = os.environ.get(rank_name)
    if rank_value is None:
        raise RuntimeError(
            f"launcher=multi 时必须在启动 bcd_search.py 前设置 {rank_name}，"
            "否则无法判断哪个节点负责写 bcd_history.json。"
        )

    controller_node_rank = int(config.get('controller_node_rank', 0))
    current_node_rank = int(rank_value)
    return current_node_rank == controller_node_rank, current_node_rank


def bcd_search(config, dry_run=False):
    """
    BCD 超参数搜索

    流程:
    1. 按 bcd_order 逐个超参数，遍历其 grid
    2. 每个 grid point: 固定其他超参不变，跑完整训练，取 final val_loss
    3. 选 loss 最低的值作为该超参最优值
    4. 一轮搜完所有超参后，如果没有任何改进则收敛
    5. 否则开始下一轮，最多 max_rounds 轮
    """
    # ---- 解析配置 ----
    optimizer_name = config['optimizer']
    base_args = config.get('base_args', {})
    base_args['optimizer'] = optimizer_name
    hyperparams = config['hyperparams']       # 各超参的搜索 grid
    bcd_order = config['bcd_order']           # 搜索顺序
    max_rounds = config.get('max_rounds', 3)
    convergence_threshold = config.get('convergence_threshold', 3e-3)
    nproc = config.get('nproc', 16)
    launcher = config.get('launcher', 'single')
    is_metadata_writer, current_node_rank = get_metadata_writer(config, launcher)
    should_write_metadata = is_metadata_writer and not dry_run
    result_timeout = config.get('result_json_timeout', 300)
    result_interval = config.get('result_json_interval', 2)

    # 输出根目录
    output_dir = config.get('output_dir', f'bcd_results/{optimizer_name}')
    runs_dir = os.path.join(output_dir, 'runs')
    if not dry_run:
        os.makedirs(runs_dir, exist_ok=True)
    control_dir = start_or_join_control_session(
        output_dir, launcher, is_metadata_writer, dry_run,
        timeout=config.get('control_timeout', 300),
        interval=config.get('control_interval', 2),
    )

    # 初始化当前最优超参（使用 defaults 或 grid 第一个值）
    current_best = {}
    for param_name in bcd_order:
        if param_name in config.get('defaults', {}):
            current_best[param_name] = config['defaults'][param_name]
        else:
            current_best[param_name] = hyperparams[param_name][0]

    # defaults 中不在 bcd_order 里的参数作为固定值，必须传给 train_gpt2.py，
    # 否则会静默回落到 train_gpt2.py 自身的 argparse default（如 warmup_fraction=0）。
    bcd_param_set = set(bcd_order)
    fixed_params = {k: v for k, v in config.get('defaults', {}).items()
                    if k not in bcd_param_set}
    if fixed_params:
        print(f"[固定参数] 不在 bcd_order 中，直接传给训练: {fixed_params}")

    # 加载已完成的 runs（用完整超参组合去重）。
    # multi 模式下只有 metadata writer 读取历史；非 0 节点通过 rank0 的决策文件跟随 run/skip。
    if launcher == 'multi' and not is_metadata_writer and not dry_run:
        completed = {}
    else:
        completed, _ = load_history(output_dir)

    print(f"\n{'#'*80}")
    print(f"BCD 超参数搜索")
    print(f"  优化器: {optimizer_name}")
    print(f"  搜索顺序: {bcd_order}")
    print(f"  最大轮数: {max_rounds}")
    print(f"  收敛阈值: {convergence_threshold}")
    print(f"  启动方式: {launcher}")
    print(f"  dry_run: {dry_run}")
    if launcher == 'multi':
        print(f"  当前 node_rank: {current_node_rank}")
        print(f"  写搜索元数据: {should_write_metadata}")
        print(f"  控制目录: {control_dir}")
    print(f"  初始值: {current_best}")
    print(f"  已完成: {len(completed)} 个不同配置")
    print(f"{'#'*80}\n")

    current_best_loss = float('inf')
    task_idx = 0

    for round_idx in range(1, max_rounds + 1):
        print(f"\n{'='*80}")
        print(f"第 {round_idx}/{max_rounds} 轮 BCD 搜索")
        print(f"{'='*80}")

        round_improved = False  # 本轮是否有任何超参的最优值发生变化

        for param_name in bcd_order:
            grid = hyperparams[param_name]
            print(f"\n--- 搜索 {param_name}: {grid} ---")
            print(f"    固定超参: {current_best}")

            old_value = current_best[param_name]
            old_params = {**fixed_params, **current_best}   # 必须含 fixed_params，否则 old_key 与 run_key 永不匹配
            old_key = params_to_key(old_params)
            old_loss = completed.get(old_key)
            ordered_grid = [old_value] + [value for value in grid if value != old_value]

            if old_value not in grid:
                print(f"    当前值 {param_name}={old_value} 不在 grid 中，将先作为 baseline 评估")

            param_best_loss = float('inf')
            param_best_value = old_value

            for value in ordered_grid:
                task_idx += 1
                # 构建本次 run 的完整超参组合：fixed_params 优先级最低，current_best 覆盖，搜索值覆盖最高
                run_params = {**fixed_params, **current_best, param_name: value}
                run_key = params_to_key(run_params)
                run_dir_name = params_to_dir_name(run_params)
                run_dir = os.path.join(runs_dir, run_dir_name)

                if launcher == 'multi' and not dry_run:
                    path = decision_path(control_dir, task_idx)
                    if is_metadata_writer:
                        cache_hit = run_key in completed
                        decision = {
                            'task_idx': task_idx,
                            'round': round_idx,
                            'param_name': param_name,
                            'param_value': value,
                            'action': 'skip' if cache_hit else 'run',
                            'val_loss': completed[run_key] if cache_hit else None,
                            'all_params': run_params,
                            'run_dir': run_dir,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        write_json_atomic(path, decision)
                    else:
                        decision = wait_for_json(
                            path,
                            timeout=config.get('control_timeout', 300),
                            interval=config.get('control_interval', 2),
                        )
                    validate_decision(decision, task_idx, round_idx, param_name, value, run_params)
                    should_run = decision['action'] == 'run'
                    val_loss = decision.get('val_loss')
                else:
                    should_run = run_key not in completed
                    val_loss = completed.get(run_key)

                if not should_run:
                    if val_loss is None:
                        raise RuntimeError(f"跳过 {param_name}={value} 但缺少 val_loss: {run_params}")
                    completed[run_key] = val_loss
                    print(f"    [跳过] {param_name}={value} -> val_loss={val_loss:.4f} (已有结果)")
                else:
                    # 构建并执行训练命令
                    cmd = build_train_command(base_args, run_params, run_dir, nproc, launcher)
                    val_loss = run_training(
                        cmd, run_dir, dry_run,
                        result_timeout=result_timeout,
                        result_interval=result_interval,
                    )

                    if dry_run:
                        val_loss = 999.0
                    elif val_loss is None:
                        raise RuntimeError(
                            f"训练失败，停止 BCD 搜索，避免写出不完整的 final_result.json: "
                            f"round={round_idx}, param={param_name}, value={value}, "
                            f"run_dir={run_dir}"
                        )

                    if run_key == old_key:
                        old_loss = val_loss
                    loss_delta_vs_old = None if old_loss is None else old_loss - val_loss

                    # 记录到历史
                    entry = {
                        'round': round_idx,
                        'param_name': param_name,
                        'param_value': value,
                        'val_loss': val_loss,
                        'old_param_value': old_value,
                        'old_loss': old_loss,
                        'loss_delta_vs_old': loss_delta_vs_old,
                        'old_params': old_params.copy(),
                        'all_params': run_params,
                        'run_dir': run_dir,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    }
                    if should_write_metadata:
                        save_history_entry(output_dir, entry)
                    completed[run_key] = val_loss

                    if loss_delta_vs_old is None:
                        print(f"    {param_name}={value} -> val_loss={val_loss:.4f}")
                    else:
                        print(f"    {param_name}={value} -> val_loss={val_loss:.4f}, "
                              f"Δvs_old={loss_delta_vs_old:.4f}")

                if run_key == old_key:
                    old_loss = val_loss

                # 记录该超参 grid 内的最优值
                if val_loss < param_best_loss:
                    param_best_loss = val_loss
                    param_best_value = value

            # 更新 current_best
            # 只有 loss 下降超过 convergence_threshold，才采用该超参的新值
            if old_loss is None:
                raise RuntimeError(
                    f"{param_name} 的 baseline 配置训练失败或缺少结果，"
                    f"无法比较 grid: {old_params}"
                )
            if param_best_loss == float('inf'):
                raise RuntimeError(f"{param_name} 的所有 grid point 都训练失败，无法继续 BCD 搜索")

            loss_delta = old_loss - param_best_loss
            print(f"\n    最优: {param_name}={param_best_value} "
                  f"(loss={param_best_loss:.4f}, Δ={loss_delta:.4f}, "
                  f"阈值={convergence_threshold})")
            if loss_delta > convergence_threshold:
                current_best[param_name] = param_best_value
                current_best_loss = param_best_loss
                round_improved = True
                print(f"    → 有实质性改进，采用 {param_name}={param_best_value} "
                      f"(原值: {old_value})")
            else:
                current_best_loss = old_loss
                print(f"    → 改进未超过阈值，保留 {param_name}={old_value}")

        # 保存本轮结果
        round_result = {
            'round': round_idx,
            'best_params': current_best.copy(),
            'best_val_loss': current_best_loss,
            'improved': round_improved,
        }
        if should_write_metadata:
            write_json_atomic(os.path.join(output_dir, f'round_{round_idx}_result.json'), round_result)

        print(f"\n第 {round_idx} 轮结束:")
        print(f"  最优超参: {current_best}")
        print(f"  最优 loss: {current_best_loss:.4f}")
        print(f"  本轮改进: {round_improved}")

        if not round_improved:
            print(f"\n所有超参数已收敛，提前结束搜索")
            break

    # 保存最终结果
    final_result = {
        'optimizer': optimizer_name,
        'best_params': current_best,
        'best_val_loss': current_best_loss,
        'total_rounds': round_idx,
        'base_args': base_args,
    }
    if should_write_metadata:
        write_json_atomic(os.path.join(output_dir, 'final_result.json'), final_result)

    print(f"\n{'#'*80}")
    print(f"BCD 搜索完成!")
    print(f"  最优超参: {current_best}")
    print(f"  最优 val_loss: {current_best_loss:.4f}")
    print(f"  结果目录: {output_dir}")
    print(f"{'#'*80}")

    return final_result


# =============================================================================
# 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BCD 超参数搜索')
    parser.add_argument('--config', type=str, required=True,
                        help='搜索配置 YAML 文件路径')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印命令，不执行训练')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    bcd_search(config, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
