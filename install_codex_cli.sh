#!/bin/bash
set -e

echo "=== Codex CLI 安装脚本 (支持 openEuler / Ubuntu / CentOS, x86_64 & aarch64) ==="

NODE_VER="22.16.0"
ARCH=$(uname -m)

case "$ARCH" in
    x86_64)  NODE_ARCH="x64" ;;
    aarch64) NODE_ARCH="arm64" ;;
    *)       echo "错误: 不支持的架构 $ARCH"; exit 1 ;;
esac

NODE_DISTRO="node-v${NODE_VER}-linux-${NODE_ARCH}"
NODE_URL="https://nodejs.org/dist/v${NODE_VER}/${NODE_DISTRO}.tar.xz"
INSTALL_DIR="/usr/local"

NEED_NODE=true
if command -v node &> /dev/null; then
    CURRENT=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$CURRENT" -ge 22 ]; then
        echo "[1/3] Node.js 已安装: $(node --version)，跳过"
        NEED_NODE=false
    else
        echo "[1/3] Node.js 版本过低 ($(node --version))，需要 >= 22，正在升级..."
    fi
else
    echo "[1/3] Node.js 未找到，正在安装 v${NODE_VER}..."
fi

if [ "$NEED_NODE" = true ]; then
    echo "     下载 ${NODE_URL} ..."
    curl -fsSL "$NODE_URL" -o /tmp/${NODE_DISTRO}.tar.xz
    echo "     解压到 ${INSTALL_DIR} ..."
    tar -xJf /tmp/${NODE_DISTRO}.tar.xz -C /tmp/
    cp -r /tmp/${NODE_DISTRO}/bin/*     ${INSTALL_DIR}/bin/
    cp -r /tmp/${NODE_DISTRO}/lib/*     ${INSTALL_DIR}/lib/
    cp -r /tmp/${NODE_DISTRO}/include/* ${INSTALL_DIR}/include/ 2>/dev/null || true
    cp -r /tmp/${NODE_DISTRO}/share/*   ${INSTALL_DIR}/share/   2>/dev/null || true
    rm -rf /tmp/${NODE_DISTRO} /tmp/${NODE_DISTRO}.tar.xz
    hash -r
fi

echo "[2/3] Node.js $(node --version) | npm $(npm --version)"

CODEX_VER="0.133"

echo "[3/3] 正在安装 @openai/codex@${CODEX_VER} ..."
npm install -g "@openai/codex@${CODEX_VER}"

echo ""
echo "=== 安装完成 ==="
echo "Codex CLI 版本: $(codex --version 2>/dev/null || echo '请运行 codex --help 验证')"
echo "npm 安装版本:"
npm list -g @openai/codex --depth=0 2>/dev/null || true

echo ""
echo "=== 安装完成 ==="
echo "Codex CLI 版本: $(codex --version 2>/dev/null || echo '请运行 codex --help 验证')"
echo ""
echo "使用方法:"
echo "  codex \"你的提示词\""
echo ""
echo "注意: 请确保设置了 OPENAI_API_KEY 环境变量"
echo "  export OPENAI_API_KEY=your-api-key"
