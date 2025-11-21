#!/bin/bash
# torch_scatter CUDA対応版インストールスクリプト
# RTX 50シリーズ(sm_120)およびその他のGPUに対応

set -e  # エラーで停止

echo "=========================================="
echo "torch_scatter CUDA対応版インストール"
echo "=========================================="

# Pythonバージョン確認
PYTHON_CMD=${PYTHON_CMD:-python}
echo "使用するPythonコマンド: $PYTHON_CMD"

# PyTorchバージョンとCUDAバージョンを確認
echo ""
echo "現在のPyTorch環境を確認中..."
PYTORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
CUDA_VERSION=$($PYTHON_CMD -c "import torch; print(torch.version.cuda if torch.version.cuda else 'cpu')" 2>/dev/null || echo "")
CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ -z "$PYTORCH_VERSION" ]; then
    echo "エラー: PyTorchがインストールされていません"
    echo "先にPyTorchをインストールしてください："
    echo "  pip install torch>=2.7.0 --index-url https://download.pytorch.org/whl/cu128"
    exit 1
fi

echo "PyTorchバージョン: $PYTORCH_VERSION"
echo "CUDAバージョン: $CUDA_VERSION"
echo "CUDA利用可能: $CUDA_AVAILABLE"

# CUDAバージョンに基づいてホイールURLを決定
if [ "$CUDA_VERSION" = "cpu" ]; then
    echo ""
    echo "警告: PyTorchがCPU版です。GPU加速を使用する場合は、CUDA版のPyTorchをインストールしてください。"
    WHEEL_URL="https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+cpu.html"
else
    # CUDAバージョンを cu形式に変換 (例: 12.8 -> cu128)
    CUDA_SHORT=$(echo $CUDA_VERSION | sed 's/\.//g')
    WHEEL_URL="https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+cu${CUDA_SHORT}.html"
fi

echo ""
echo "インストールURL: $WHEEL_URL"

# 既存のtorch_scatterをアンインストール
echo ""
echo "既存のtorch_scatterをアンインストール中..."
$PYTHON_CMD -m pip uninstall -y torch-scatter torch-sparse torch-cluster 2>/dev/null || true

# キャッシュをクリア
echo "pipキャッシュをクリア中..."
$PYTHON_CMD -m pip cache purge 2>/dev/null || true

# CUDA対応版をインストール
echo ""
echo "torch_scatter CUDA対応版をインストール中..."
$PYTHON_CMD -m pip install --no-cache-dir torch-scatter torch-sparse torch-cluster -f "$WHEEL_URL"

# インストール確認
echo ""
echo "=========================================="
echo "インストール確認"
echo "=========================================="

$PYTHON_CMD -c "
import torch
import torch_scatter

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'torch_scatter: {torch_scatter.__version__}')

# CUDA機能のテスト
if torch.cuda.is_available():
    print('')
    print('CUDA機能テスト中...')
    try:
        x = torch.randn(10, 3).cuda()
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3]).cuda()
        result = torch_scatter.scatter_max(x, batch, dim=0)
        print('✓ torch_scatter CUDA機能が正常に動作しています')
    except Exception as e:
        print(f'✗ CUDA機能テストに失敗: {e}')
        exit(1)
else:
    print('')
    print('警告: CUDAが利用できません。CPU版として動作します。')
"

echo ""
echo "=========================================="
echo "インストール完了"
echo "=========================================="
