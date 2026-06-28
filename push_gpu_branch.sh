#!/usr/bin/env bash
# push_gpu_branch.sh — 将当前目录推送为 GitHub 上的 GPU_version 分支
# 可重复执行，每次运行会自动 commit 变更并 push

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BRANCH="GPU_version"
REMOTE="origin"

# 允许 git 操作当前目录（多用户环境下的 safe.directory 问题）
git config --global --add safe.directory "$REPO_DIR" 2>/dev/null || true

cd "$REPO_DIR"

# 确认 remote 指向正确
echo "[1/5] Remote:"
git remote get-url "$REMOTE"

# 切换到 GPU_version 分支（不存在则从当前状态新建）
if ! git show-ref --quiet refs/heads/"$BRANCH"; then
    echo "[2/5] Branch '$BRANCH' not found locally, creating..."
    git checkout -b "$BRANCH"
fi
git checkout "$BRANCH"

echo "[2/5] On branch: $(git branch --show-current)"

# 暂存所有变更（.gitignore 会自动排除 pretraining_record/ 等）
git add -A

# 只有有变更时才 commit
if git diff --cached --quiet; then
    echo "[3/5] Nothing to commit, working tree clean."
else
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    git commit -m "GPU_version update: $TIMESTAMP"
    echo "[3/5] Committed."
fi

# Push（首次用 -u 设置 upstream，后续自动）
echo "[4/5] Pushing to $REMOTE/$BRANCH ..."
git push -u "$REMOTE" "$BRANCH"

echo "[5/5] Done. Branch '$BRANCH' pushed to $(git remote get-url $REMOTE)"
