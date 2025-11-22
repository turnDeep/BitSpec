#!/bin/bash
# コンテナ内で実行するtorch_scatter修正スクリプト

set -e

echo "=========================================="
echo "torch_scatter CUDA対応修正スクリプト"
echo "=========================================="

# 1. 現在の環境確認
echo -e "\n[1/5] 現在の環境確認"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 2. 既存のtorch_scatterを削除
echo -e "\n[2/5] 既存のtorch_scatterを削除"
pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv 2>/dev/null || echo "既存パッケージなし"

# 3. PyTorchバージョンを検出
echo -e "\n[3/5] PyTorchバージョンを検出"
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
echo "PyTorch version: ${TORCH_VERSION}"
echo "CUDA version: ${CUDA_VERSION}"

# 4. ビルド済みホイールからインストールを試行
echo -e "\n[4/5] ビルド済みホイールからインストール"
WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html"
echo "Wheel URL: ${WHEEL_URL}"

pip install --no-cache-dir --no-build-isolation \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f ${WHEEL_URL} || {
    echo "⚠️  ビルド済みホイールが見つかりません。ソースからビルドします..."

    # ninja（高速ビルドツール）がインストールされているか確認
    pip install ninja

    # ソースからビルド（CUDA対応を強制）
    TORCH_CUDA_ARCH_LIST="9.0;12.0" \
    FORCE_CUDA=1 \
    pip install --no-cache-dir --no-build-isolation \
        torch-scatter torch-sparse torch-cluster torch-spline-conv
}

# 5. インストール確認
echo -e "\n[5/5] インストール確認"
python -c "
import torch
import torch_scatter

print(f'✅ torch_scatter version: {torch_scatter.__version__}')
print(f'✅ torch_scatter location: {torch_scatter.__file__}')

# CUDA演算テスト
if torch.cuda.is_available():
    src = torch.randn(10, 5).cuda()
    index = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 0, 1]).cuda()
    out = torch_scatter.scatter(src, index, dim=0, reduce='sum')
    print('✅ torch_scatter CUDA演算テスト成功!')
else:
    print('⚠️  CUDA利用不可')
"

echo -e "\n=========================================="
echo "✅ 修正完了"
echo "=========================================="
