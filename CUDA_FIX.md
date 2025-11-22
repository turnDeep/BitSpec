# torch-scatter CUDA Support Fix

## Problem

The training pipeline fails with:
```
RuntimeError: Not compiled with CUDA support
```

This occurs because `torch-scatter` and related PyTorch Geometric extensions were not installed with CUDA support. This is common when using PyTorch nightly builds, as pre-built wheels may not be available.

## Quick Fix (Recommended)

Run the following commands **inside the devcontainer terminal**:

### Option 1: Using the fix script
```bash
./fix_torch_scatter.sh
```

### Option 2: Manual installation
```bash
# If using /opt/venv (check Dockerfile setup)
source /opt/venv/bin/activate

# Uninstall CPU-only versions
pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv

# Install from source with CUDA support
pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_scatter.git
pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_sparse.git
pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_cluster.git
pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_spline_conv.git

# Verify installation
python -c "import torch; import torch_scatter; print('✅ CUDA support:', torch.cuda.is_available())"
```

## Permanent Fix

The Dockerfile has been updated to build PyG extensions from source. To apply this permanently:

1. Rebuild the devcontainer:
   - In VS Code: Press `F1` → "Dev Containers: Rebuild Container"
   - Or manually: `docker-compose down && docker-compose up --build`

2. The updated Dockerfile now includes:
   ```dockerfile
   # Build PyG extensions from source for CUDA support
   RUN pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_scatter.git
   RUN pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_sparse.git
   RUN pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_cluster.git
   RUN pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_spline_conv.git
   ```

## Verification

After installation, verify CUDA support:

```bash
python -c "
import torch
import torch_scatter

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    # Test torch_scatter with CUDA
    src = torch.randn(10, 5).cuda()
    index = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 0, 1]).cuda()
    out = torch_scatter.scatter(src, index, dim=0, reduce='sum')
    print('✅ torch_scatter CUDA test passed!')
"
```

## Additional Warnings (Non-Critical)

You may also see these deprecation warnings, which are non-critical:

1. **RDKit MorganGenerator**:
   - Warning: `DEPRECATION WARNING: please use MorganGenerator`
   - This is a deprecation warning from RDKit and doesn't affect functionality

2. **PyTorch AMP**:
   - `torch.cuda.amp.GradScaler` → use `torch.amp.GradScaler('cuda')`
   - `torch.cuda.amp.autocast` → use `torch.amp.autocast('cuda')`
   - These can be fixed in the code later but don't cause failures

## Notes

- Building from source takes 5-10 minutes per library
- Requires CUDA development tools (already included in the Docker image)
- RTX 50 series (sm_120) support requires PyTorch nightly build
