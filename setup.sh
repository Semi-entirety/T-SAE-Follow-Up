#!/bin/bash
# setup.sh - RTX4090 24GB 环境配置脚本
# 用法: bash setup.sh

set -e  # 任何步骤报错就停止

echo "=============================="
echo " T-SAE 环境配置开始"
echo "=============================="

# ── 1. 克隆仓库 ──────────────────────────────────────────────
echo "[1/6] 克隆项目仓库..."
git clone https://github.com/Semi-entirety/T-SAE-Follow-Up.git
cd T-SAE-Follow-Up

# ── 2. 安装 Miniconda（如果没有）────────────────────────────
echo "[2/6] 检查 Conda..."
if ! command -v conda &> /dev/null; then
    echo "Conda 未找到，正在安装 Miniconda..."
    wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b
    rm miniconda.sh
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc
    echo "Miniconda 安装完成"
else
    echo "Conda 已存在，跳过安装"
fi

# ── 3. 创建 conda 环境 ───────────────────────────────────────
echo "[3/6] 创建 conda 环境 t-sae (Python 3.11)..."
~/miniconda3/bin/conda create -n t-sae python=3.11 -y
source ~/miniconda3/bin/activate t-sae

# ── 4. 安装 temporal-saes ────────────────────────────────────
echo "[4/6] 安装 temporal-saes..."
cd temporal-saes
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..

# ── 5. 安装项目依赖 ──────────────────────────────────────────
echo "[5/6] 安装项目依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── 6. 验证环境 ──────────────────────────────────────────────
echo "[6/6] 验证环境..."
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "=============================="
echo " 环境配置完成！"
echo "=============================="
echo ""
echo "激活环境："
echo "  source ~/miniconda3/bin/activate t-sae"
echo ""
echo "开始训练（RTX4090 推荐参数）："
echo "  python spatial_train.py \\"
echo "      --image_root /path/to/imagenet \\"
echo "      --dino_model dinov2_vitb14 \\"
echo "      --batch_size_images 16 \\"
echo "      --pairs_per_image 128 \\"
echo "      --dict_size 16384 \\"
echo "      --contrastive_alpha 3.0 \\"
echo "      --steps 100000 \\"
echo "      --device cuda \\"
echo "      --save_dir ./checkpoints_imagenet"
