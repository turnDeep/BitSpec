#!/bin/bash
# Fix torch-scatter CUDA support issue
# This script must be run inside the devcontainer

set -e

echo "=========================================="
echo "Fixing torch-scatter CUDA support"
echo "=========================================="

# Check if running in the correct environment
if [ ! -f "/opt/venv/bin/python" ]; then
    echo "Error: This script must be run inside the devcontainer"
    echo "Please run this script from within VS Code devcontainer"
    exit 1
fi

# Activate virtual environment
source /opt/venv/bin/activate

# Check current PyTorch version
echo ""
echo "Current PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Uninstall existing PyG extensions
echo ""
echo "Uninstalling existing PyG extensions..."
pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv 2>/dev/null || true

# Install from source with CUDA support
echo ""
echo "Installing torch-scatter from source (this may take a few minutes)..."
pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_scatter.git

echo ""
echo "Installing torch-sparse from source..."
pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_sparse.git

echo ""
echo "Installing torch-cluster from source..."
pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_cluster.git

echo ""
echo "Installing torch-spline-conv from source..."
pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_spline_conv.git

# Verify installation
echo ""
echo "=========================================="
echo "Verifying CUDA support..."
echo "=========================================="
python -c "
import torch
import torch_scatter

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'torch_scatter installed: {torch_scatter.__version__}')

if torch.cuda.is_available():
    print('')
    print('Testing torch_scatter CUDA operations...')
    src = torch.randn(10, 5).cuda()
    index = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 0, 1]).cuda()
    out = torch_scatter.scatter(src, index, dim=0, reduce='sum')
    print('✅ torch_scatter CUDA test passed!')
else:
    print('⚠️  CUDA not available')
"

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "You can now run the training pipeline:"
echo "  python scripts/train_pipeline.py --config config.yaml"
