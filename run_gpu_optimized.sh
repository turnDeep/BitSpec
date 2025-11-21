#!/bin/bash
# GPU最適化実行スクリプト
# RTX 40/50シリーズ向けの最大性能設定

set -e  # エラーで停止

echo "=========================================="
echo "NEIMS v2.0 GPU最適化実行"
echo "=========================================="

# ========================================
# GPU環境変数設定
# ========================================

# 使用するGPUを指定（複数GPU: "0,1,2,3"）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "使用GPU: $CUDA_VISIBLE_DEVICES"

# CUDAアーキテクチャリスト
# RTX 40シリーズ (Ada Lovelace): 8.9
# RTX 50シリーズ (Blackwell): 12.0
# 両方サポート: "8.9;9.0;12.0"
export TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"
echo "CUDAアーキテクチャ: $TORCH_CUDA_ARCH_LIST"

# CUDAキャッシュ設定
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/tmp/cuda_cache}
mkdir -p $CUDA_CACHE_PATH
echo "CUDAキャッシュ: $CUDA_CACHE_PATH (最大2GB)"

# cuDNN最適化
export CUDNN_BENCHMARK=1              # 最速アルゴリズムを自動選択
export CUDNN_DETERMINISTIC=0          # 再現性より速度優先
echo "cuDNNベンチマーク: 有効"

# OpenMP設定（CPUスレッド数）
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
echo "CPUスレッド数: $OMP_NUM_THREADS"

# PyTorch設定
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
echo "CUDAメモリアロケーター: max_split_size_mb=512"

# ========================================
# GPU情報表示
# ========================================

echo ""
echo "=========================================="
echo "GPU情報"
echo "=========================================="

PYTHON_CMD=${PYTHON_CMD:-python}

$PYTHON_CMD -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}')
        print(f'  - Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'  - Compute Capability: {props.major}.{props.minor}')
        print(f'  - Multi-Processors: {props.multi_processor_count}')
" || {
    echo "エラー: PyTorchまたはCUDAが正しく設定されていません"
    exit 1
}

# ========================================
# torch_scatterの確認
# ========================================

echo ""
echo "=========================================="
echo "torch_scatter確認"
echo "=========================================="

$PYTHON_CMD -c "
import torch
import torch_scatter

print(f'torch_scatter: {torch_scatter.__version__}')

# CUDA機能テスト
if torch.cuda.is_available():
    try:
        x = torch.randn(10, 3).cuda()
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3]).cuda()
        result = torch_scatter.scatter_max(x, batch, dim=0)
        print('✓ torch_scatter CUDA機能が正常に動作しています')
    except Exception as e:
        print(f'✗ torch_scatter CUDA機能テストに失敗: {e}')
        print('')
        print('エラー対処法:')
        print('  bash install_torch_scatter_cuda.sh')
        exit(1)
else:
    print('警告: CUDAが利用できません')
" || {
    echo ""
    echo "エラー: torch_scatterがCUDAサポートでインストールされていません"
    echo "以下のコマンドを実行してください:"
    echo "  bash install_torch_scatter_cuda.sh"
    exit 1
}

# ========================================
# 設定ファイル選択
# ========================================

echo ""
echo "=========================================="
echo "設定ファイル選択"
echo "=========================================="

CONFIG_FILE=${CONFIG_FILE:-config_gpu_optimized.yaml}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "エラー: 設定ファイルが見つかりません: $CONFIG_FILE"
    exit 1
fi

echo "使用する設定: $CONFIG_FILE"

# GPUメモリに応じたバッチサイズの推奨
echo ""
echo "バッチサイズ推奨値（GPUメモリ別）:"
echo "  16GB (RTX 4060 Ti, RTX 5070 Ti): batch_size: 128"
echo "  24GB (RTX 4090, RTX 5080): batch_size: 256"
echo "  32GB+ (RTX 5090): batch_size: 512"
echo ""
echo "config_gpu_optimized.yamlで調整してください"

# ========================================
# 訓練実行
# ========================================

echo ""
echo "=========================================="
echo "訓練開始"
echo "=========================================="

# 開始フェーズ
START_PHASE=${START_PHASE:-1}

# デバイス
DEVICE=${DEVICE:-cuda}

# 追加引数
EXTRA_ARGS=${EXTRA_ARGS:-}

# 実行コマンド
CMD="$PYTHON_CMD scripts/train_pipeline.py --config $CONFIG_FILE --start-phase $START_PHASE --device $DEVICE $EXTRA_ARGS"

echo "実行コマンド:"
echo "  $CMD"
echo ""

# タイムスタンプ記録
START_TIME=$(date +%s)

# 訓練実行
$CMD

# 終了時刻計算
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "訓練完了"
echo "=========================================="
echo "経過時間: ${HOURS}時間 ${MINUTES}分 ${SECONDS}秒"

# GPU使用状況の最終確認
echo ""
echo "GPU使用状況:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || echo "nvidia-smiが利用できません"
