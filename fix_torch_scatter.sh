#!/bin/bash
# torch_scatter CUDA対応版を自動的にインストールするスクリプト
# エラー "RuntimeError: Not compiled with CUDA support" を修正

set -e

echo "=========================================="
echo "torch_scatter CUDA修正スクリプト"
echo "=========================================="

# Pythonコマンドを検出
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "エラー: Pythonが見つかりません"
    exit 1
fi

echo "使用するPython: $PYTHON_CMD ($($PYTHON_CMD --version))"

# PyTorchとCUDAバージョンを確認
echo ""
echo "PyTorch環境を確認中..."

PYTORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
if [ -z "$PYTORCH_VERSION" ]; then
    echo "エラー: PyTorchがインストールされていません"
    echo "先にPyTorchをインストールしてください"
    exit 1
fi

CUDA_VERSION=$($PYTHON_CMD -c "import torch; print(torch.version.cuda if torch.version.cuda else 'cpu')" 2>/dev/null)
CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)

echo "PyTorchバージョン: $PYTORCH_VERSION"
echo "PyTorch CUDAバージョン: $CUDA_VERSION"
echo "CUDA利用可能: $CUDA_AVAILABLE"

# PyTorchのCUDAバリアントを確認
if [[ $PYTORCH_VERSION == *"+"* ]]; then
    PYTORCH_CUDA_SUFFIX=$(echo $PYTORCH_VERSION | cut -d'+' -f2)
    echo "PyTorch CUDAサフィックス: $PYTORCH_CUDA_SUFFIX"
else
    # サフィックスがない場合、CUDAバージョンから推測
    if [ "$CUDA_VERSION" = "cpu" ] || [ "$CUDA_AVAILABLE" = "False" ]; then
        PYTORCH_CUDA_SUFFIX="cpu"
    elif [[ "$CUDA_VERSION" == 12.8* ]]; then
        PYTORCH_CUDA_SUFFIX="cu128"
    elif [[ "$CUDA_VERSION" == 12.6* ]]; then
        PYTORCH_CUDA_SUFFIX="cu126"
    elif [[ "$CUDA_VERSION" == 12.4* ]]; then
        PYTORCH_CUDA_SUFFIX="cu124"
    elif [[ "$CUDA_VERSION" == 12.1* ]]; then
        PYTORCH_CUDA_SUFFIX="cu121"
    elif [[ "$CUDA_VERSION" == 11.8* ]]; then
        PYTORCH_CUDA_SUFFIX="cu118"
    elif [[ "$CUDA_VERSION" == 11.7* ]]; then
        PYTORCH_CUDA_SUFFIX="cu117"
    else
        # デフォルトでcu118を使用（最も互換性が高い）
        echo "警告: CUDAバージョンを検出できません。cu118を使用します"
        PYTORCH_CUDA_SUFFIX="cu118"
    fi
fi

echo "使用するCUDAサフィックス: $PYTORCH_CUDA_SUFFIX"

# PyTorchのメジャーバージョンを取得（2.7.0 -> 2.7.0）
PYTORCH_BASE_VERSION=$PYTORCH_VERSION

# Wheelページを構築
WHEEL_URL="https://data.pyg.org/whl/torch-${PYTORCH_BASE_VERSION}+${PYTORCH_CUDA_SUFFIX}.html"

echo ""
echo "インストールURL: $WHEEL_URL"

# 既存のtorch_scatterを確認
echo ""
echo "現在のtorch_scatter状態を確認中..."
$PYTHON_CMD -c "
import sys
try:
    import torch_scatter
    print(f'torch_scatter {torch_scatter.__version__} がインストール済み')
    print('アンインストールして再インストールします')
except ImportError:
    print('torch_scatterはインストールされていません')
" 2>/dev/null || echo "torch_scatterはインストールされていません"

# 既存のパッケージをアンインストール
echo ""
echo "既存のPyG拡張をアンインストール中..."
$PYTHON_CMD -m pip uninstall -y torch-scatter torch-sparse torch-cluster 2>/dev/null || true

# pipキャッシュをクリア
echo "pipキャッシュをクリア中..."
$PYTHON_CMD -m pip cache purge 2>/dev/null || echo "キャッシュクリアをスキップ"

# CUDA対応版をインストール
echo ""
echo "torch_scatter CUDA対応版をインストール中..."
echo "コマンド: pip install --no-cache-dir torch-scatter torch-sparse torch-cluster -f \"$WHEEL_URL\""

$PYTHON_CMD -m pip install --no-cache-dir torch-scatter torch-sparse torch-cluster -f "$WHEEL_URL"

# インストール確認とテスト
echo ""
echo "=========================================="
echo "インストール確認"
echo "=========================================="

$PYTHON_CMD -c "
import sys
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
        print('✓ torch_scatter CUDA機能が正常に動作しています!')
        print('✓ RuntimeError: Not compiled with CUDA support エラーは修正されました')
        sys.exit(0)
    except Exception as e:
        print(f'✗ CUDA機能テストに失敗: {e}')
        print('')
        print('トラブルシューティング:')
        print('1. PyTorchとCUDAのバージョンを確認してください')
        print('2. 以下のコマンドで手動インストールを試してください:')
        print(f'   pip install --force-reinstall --no-cache-dir torch-scatter -f {WHEEL_URL}')
        sys.exit(1)
else:
    print('')
    print('警告: CUDAが利用できません')
    print('GPU加速を使用する場合は、CUDA対応版のPyTorchをインストールしてください')
    print('')
    print('CUDA対応版PyTorchのインストール:')
    print('  pip install torch>=2.7.0 --index-url https://download.pytorch.org/whl/cu128')
    sys.exit(0)
"

RESULT=$?

echo ""
echo "=========================================="
if [ $RESULT -eq 0 ]; then
    echo "✓ インストール成功!"
else
    echo "✗ インストールに問題が発生しました"
fi
echo "=========================================="

exit $RESULT
