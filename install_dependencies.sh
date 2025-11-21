#!/bin/bash
# install_dependencies.sh
# NExtIMS 依存関係インストールスクリプト
# torch-scatter の CUDA サポートを確実にインストールします

set -e  # エラーで停止

echo "=========================================="
echo "NExtIMS 依存関係インストール"
echo "=========================================="

# Python バージョン確認
echo ""
echo "[1/6] Python バージョン確認..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Python 3.10+ 要件チェック
required_version="3.10"
if ! python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "エラー: Python 3.10 以降が必要です (現在: $python_version)"
    exit 1
fi

# pip アップグレード
echo ""
echo "[2/6] pip をアップグレード..."
python -m pip install --upgrade pip

# CUDA 確認
echo ""
echo "[3/6] CUDA 環境確認..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "CUDA 環境が検出されました"
else
    echo "警告: nvidia-smi が見つかりません。CUDA が利用可能か確認してください"
fi

# PyTorch インストール (CUDA 12.4)
echo ""
echo "[4/6] PyTorch 2.5.1 (CUDA 12.4) をインストール..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# PyTorch インストール確認
echo ""
echo "PyTorch インストール確認:"
python -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'  CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'  GPU count: {torch.cuda.device_count()}')"
    if python -c "import torch; exit(0 if torch.cuda.device_count() > 0 else 1)"; then
        python -c "import torch; print(f'  GPU 0: {torch.cuda.get_device_name(0)}')"
    fi
else
    echo "  警告: CUDA が利用できません。CPU モードで動作します"
fi

# torch-scatter などの PyG 拡張をインストール
echo ""
echo "[5/6] PyTorch Geometric 拡張 (CUDA 対応) をインストール..."
echo "  - torch-scatter"
echo "  - torch-sparse"
echo "  - torch-cluster"
echo "  - torch-spline-conv"
pip install torch-scatter==2.1.2+pt25cu124 \
    torch-sparse==0.6.18+pt25cu124 \
    torch-cluster==1.6.3+pt25cu124 \
    torch-spline-conv==1.2.2+pt25cu124 \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

# torch-scatter CUDA サポート確認
echo ""
echo "torch-scatter CUDA サポート確認:"
python -c "
import torch
import torch_scatter

print(f'  torch-scatter version: {torch_scatter.__version__}')

# CUDA サポートのテスト
if torch.cuda.is_available():
    try:
        x = torch.randn(10, 16).cuda()
        index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).cuda()
        out = torch_scatter.scatter_mean(x, index, dim=0)
        print('  ✓ torch-scatter CUDA サポート: 正常')
    except Exception as e:
        print(f'  ✗ torch-scatter CUDA エラー: {e}')
        exit(1)
else:
    print('  - CUDA が利用できないため、CPU モードでテスト')
    x = torch.randn(10, 16)
    index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    out = torch_scatter.scatter_mean(x, index, dim=0)
    print('  ✓ torch-scatter CPU モード: 正常')
"

# 残りの依存関係をインストール
echo ""
echo "[6/6] 残りの依存関係をインストール..."
pip install torch-geometric>=2.5.0
pip install rdkit>=2023.9.1
pip install mordred>=1.2.0
pip install mol2vec>=0.1
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install h5py>=3.10.0
pip install scipy>=1.11.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.13.0
pip install plotly>=5.18.0
pip install pillow>=10.0.0
pip install tqdm>=4.66.0
pip install pyyaml>=6.0
pip install tensorboard>=2.15.0
pip install wandb>=0.16.0
pip install torch-ema>=0.3.0

# 開発用パッケージ
echo ""
echo "開発用パッケージをインストール..."
pip install pytest>=7.4.0
pip install black>=23.12.0
pip install flake8>=7.0.0
pip install mypy>=1.8.0
pip install jupyter>=1.0.0

# パッケージ本体のインストール (editable mode)
echo ""
echo "NExtIMS パッケージをインストール..."
pip install -e .

# 最終確認
echo ""
echo "=========================================="
echo "インストール完了確認"
echo "=========================================="
python -c "
import torch
import torch_scatter
import torch_geometric
import rdkit
from rdkit import Chem

print('✓ PyTorch:', torch.__version__)
print('✓ torch-scatter:', torch_scatter.__version__)
print('✓ torch-geometric:', torch_geometric.__version__)
print('✓ RDKit:', rdkit.__version__)
print('✓ CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('✓ CUDA version:', torch.version.cuda)
    print('✓ GPU:', torch.cuda.get_device_name(0))

    # torch-scatter CUDA 最終テスト
    x = torch.randn(1000, 128).cuda()
    index = torch.randint(0, 100, (1000,)).cuda()
    out = torch_scatter.scatter_mean(x, index, dim=0)
    print('✓ torch-scatter CUDA: 動作確認完了')
"

echo ""
echo "=========================================="
echo "すべてのインストールが正常に完了しました！"
echo "=========================================="
echo ""
echo "次のステップ:"
echo "  1. データを配置: data/NIST17.msp, data/mol_files/"
echo "  2. 学習を開始: python scripts/train_pipeline.py --config config.yaml"
echo "  3. 推論を実行: python scripts/predict.py --help"
echo ""
