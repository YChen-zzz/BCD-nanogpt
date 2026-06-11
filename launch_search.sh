#!/bin/bash
# =============================================================================
# 一键启动 BCD 超参数搜索
#
# 用法:
#   bash launch_search.sh configs/search_muon.yaml        # 启动 Muon 搜索
#   bash launch_search.sh configs/search_adamw.yaml       # 启动 AdamW 搜索
#   bash launch_search.sh configs/search_muon.yaml --dry_run  # 只打印命令
# =============================================================================

if [ $# -lt 1 ]; then
    echo "用法: bash launch_search.sh <config.yaml> [--dry_run]"
    echo ""
    echo "可用配置:"
    ls configs/search_*.yaml 2>/dev/null
    exit 1
fi

CONFIG=$1
shift

echo "=========================================="
echo "启动 BCD 超参数搜索"
echo "配置文件: $CONFIG"
echo "=========================================="

python bcd_search.py --config "$CONFIG" "$@"
