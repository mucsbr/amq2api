#!/bin/bash

# Amazon Q to Claude API Proxy 启动脚本

set -e

echo "=========================================="
echo "Amazon Q to Claude API Proxy"
echo "=========================================="

# 检查 Python 版本
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python 3"
    exit 1
fi

echo "Python 版本: $(python3 --version)"

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "警告: .env 文件不存在"
    echo "请复制 .env.example 并填写配置信息："
    echo "  cp .env.example .env"
    echo "  然后编辑 .env 文件"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 启动服务
echo "=========================================="
echo "启动服务..."
echo "=========================================="
python3 main.py
