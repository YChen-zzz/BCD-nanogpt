"""
验证 DistributedDataLoader 数据读取顺序的测试脚本

验证以下事实：
  1. 文件按数字顺序加载（零填充文件名下字母序 == 数字序）
  2. 多 GPU 交错读取，各自不重叠
  3. shard 耗尽后循环回到 shard 0
  4. reset() 后数据完全重现（确定性）
"""

import sys
import glob
import re
import numpy as np
import torch
from unittest.mock import patch

sys.path.insert(0, '/data/250010186/Pretraining_code/BCD-nanogpt/BCD-nanogpt')
from train_gpt2 import DistributedDataLoader, _peek_data_shard

DATA_PATTERN = '/data/250010186/fineweb100B/cached_fineweb100B/fineweb_train_*.bin'

def sep(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ─────────────────────────────────────────────────────────────
# Test 1: 文件名排序 == 数字顺序
# ─────────────────────────────────────────────────────────────
def test_file_order():
    sep("Test 1: 文件名排序是否等于数字顺序")

    files = sorted(glob.glob(DATA_PATTERN))
    assert len(files) > 0, "未找到任何 .bin 文件"

    numbers = [int(re.search(r'(\d+)\.bin$', f).group(1)) for f in files]

    for i in range(len(numbers) - 1):
        assert numbers[i] < numbers[i+1], \
            f"顺序错误: {files[i]} -> {files[i+1]}"

    print(f"  总 shard 数: {len(files)}")
    print(f"  前 3 个: {[f.split('/')[-1] for f in files[:3]]}")
    print(f"  后 3 个: {[f.split('/')[-1] for f in files[-3:]]}")
    print(f"  编号范围: {numbers[0]:06d} ~ {numbers[-1]:06d}")
    print("  [PASS] 字母序与数字序完全一致")


# ─────────────────────────────────────────────────────────────
# Test 2: 多 GPU 交错读取，各自不重叠
# ─────────────────────────────────────────────────────────────
def test_interleaved_no_overlap():
    sep("Test 2: 多 GPU 交错读取，各自不重叠")

    B, T = 4, 1024
    num_gpus = 4

    loaders = [
        DistributedDataLoader(DATA_PATTERN, B, T, rank, num_gpus)
        for rank in range(num_gpus)
    ]

    # 验证各 GPU 初始 position
    for rank, loader in enumerate(loaders):
        expected = rank * B * T
        assert loader.current_position == expected, \
            f"GPU{rank} 初始位置错误: 期望 {expected}, 实际 {loader.current_position}"
        print(f"  GPU{rank} 初始 position = {loader.current_position}  (= {rank} × {B} × {T})")

    # 验证读取区间不重叠
    intervals = [(r * B * T, r * B * T + B * T) for r in range(num_gpus)]
    for i in range(num_gpus):
        for j in range(i + 1, num_gpus):
            lo1, hi1 = intervals[i]
            lo2, hi2 = intervals[j]
            assert max(0, min(hi1, hi2) - max(lo1, lo2)) == 0, \
                f"GPU{i} 和 GPU{j} 区间重叠"
    print(f"  [PASS] {num_gpus} 个 GPU 读取区间互不重叠")

    # 验证实际 token 与 shard 原始数据一致
    shard0_tokens = loaders[0].tokens
    for rank, loader in enumerate(loaders):
        x, y = loader.next_batch()
        x_np = x.cpu().numpy().flatten()
        expected = shard0_tokens[rank * B * T : rank * B * T + B * T].astype(np.int32)
        assert np.array_equal(x_np, expected), \
            f"GPU{rank} token 与 shard 原始数据不符"
        print(f"  GPU{rank} x[0,:4] = {x_np[:4]}  (shard 偏移 {rank*B*T})")
    print("  [PASS] 实际读取 token 与 shard 原始数据完全对应")


# ─────────────────────────────────────────────────────────────
# Test 3: shard 耗尽后循环回到 shard 0（用 mock 避免加载所有 shard）
# ─────────────────────────────────────────────────────────────
def test_shard_wraparound():
    sep("Test 3: shard 耗尽后循环回到 shard 0")

    B, T = 4, 1024
    loader = DistributedDataLoader(DATA_PATTERN, B, T, 0, 1)
    total_shards = len(loader.files)
    shard0_tokens = loader.tokens.copy()

    print(f"  总 shard 数: {total_shards}")

    # 用 mock 替代 _load_data_shard，避免真正读 852 个文件
    # mock 返回与 shard0 相同大小的假数据（只测循环逻辑）
    fake_tokens = np.zeros(len(shard0_tokens), dtype=np.uint16)

    with patch('train_gpt2._load_data_shard', return_value=fake_tokens):
        # 直接跳到最后一个 shard
        loader.current_shard = total_shards - 1
        loader.tokens = fake_tokens
        print(f"  手动设置 current_shard = {loader.current_shard}  "
              f"({loader.files[loader.current_shard].split('/')[-1]})")

        # advance 一次，应该 wrap 回 0
        loader.advance()
        print(f"  advance 后 current_shard = {loader.current_shard}  "
              f"({loader.files[loader.current_shard].split('/')[-1]})")
        assert loader.current_shard == 0, \
            f"wrap 后 shard 应为 0，实际为 {loader.current_shard}"

    # 再验证：从 shard 0 真实读取的数据与初始一致
    loader.reset()
    assert np.array_equal(loader.tokens, shard0_tokens), \
        "reset 后 shard 0 数据与初始不一致"

    print(f"  reset 后 shard 0 第一个 token: {int(loader.tokens[0])}")
    print("  [PASS] advance 后正确 wrap 回 shard 0")


# ─────────────────────────────────────────────────────────────
# Test 4: reset() 后数据完全重现
# ─────────────────────────────────────────────────────────────
def test_determinism_after_reset():
    sep("Test 4: reset() 后数据完全重现（确定性）")

    B, T = 8, 1024
    loader = DistributedDataLoader(DATA_PATTERN, B, T, 0, 1)

    first_run = []
    for _ in range(10):
        x, y = loader.next_batch()
        first_run.append(x.cpu().clone())

    loader.reset()

    for i in range(10):
        x, y = loader.next_batch()
        assert torch.equal(x.cpu(), first_run[i]), f"第 {i} 个 batch reset 后不一致"

    print(f"  读取了 10 个 batch，reset 后全部重现")
    print(f"  batch[0] x[0,:4] = {first_run[0][0,:4].tolist()}")
    print(f"  batch[9] x[0,:4] = {first_run[9][0,:4].tolist()}")
    print("  [PASS] reset() 后数据顺序完全确定")


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_file_order()
    test_interleaved_no_overlap()
    test_shard_wraparound()
    test_determinism_after_reset()
    print(f"\n{'='*60}")
    print("所有测试通过")
    print(f"{'='*60}")
