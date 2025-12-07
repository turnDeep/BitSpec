#!/bin/bash
# DGL Installation Script for CUDA 12.8 (Blackwell / sm_120)
#
# This script installs DGL from source with support for CUDA 12.8 and sm_120 architecture.
#
# Prerequisites:
# - CUDA 12.8 must be installed
# - PyTorch with CUDA support must be installed
# - CMake and build tools must be available
#
# Usage:
#   bash scripts/install_dgl_cuda_12_8.sh

set -e  # Exit on error

echo "================================================================================"
echo "DGL Installation for CUDA 12.8 (Blackwell / sm_120)"
echo "================================================================================"

# Set CUDA environment variables
export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA 12.8 is installed."
    exit 1
fi

echo "CUDA Version:"
nvcc --version

# Set DGL build directory
DGL_BUILD_DIR=/tmp/dgl_build
rm -rf $DGL_BUILD_DIR
mkdir -p $DGL_BUILD_DIR
cd $DGL_BUILD_DIR

# Clone DGL repository (shallow clone for faster download)
echo ""
echo "Cloning DGL repository..."
git clone --depth=1 https://github.com/dmlc/dgl.git
cd dgl

# Initialize required submodules only
echo ""
echo "Initializing required submodules..."
git submodule update --init --depth=1 \
    third_party/dmlc-core \
    third_party/dlpack \
    third_party/cccl \
    third_party/cuco

# Apply CUDA 12.8 / sm_120 patch
echo ""
echo "Applying CUDA 12.8 / sm_120 patch..."

# Check if patch file exists in the project
PATCH_FILE="$(dirname "$(dirname "$(readlink -f "$0")")")/patches/dgl/CUDA.cmake"
if [ -f "$PATCH_FILE" ]; then
    echo "Using patch from: $PATCH_FILE"
    cp "$PATCH_FILE" cmake/modules/CUDA.cmake
else
    echo "Patch file not found. Applying inline patch..."

    # Add sm_120 support to known GPU architectures
    sed -i '/if (CUDA_VERSION VERSION_GREATER_EQUAL "12.0")/,/endif()/ {
        /endif()/ a\
# Add support for CUDA 12.8 / sm_120 (Blackwell)\
if (CUDA_VERSION VERSION_GREATER_EQUAL "12.8")\
  list(APPEND dgl_known_gpu_archs "120")\
  set(dgl_cuda_arch_ptx "120")\
endif()
    }' cmake/modules/CUDA.cmake

    # Add Blackwell to architecture names
    sed -i 's/\(set(__archs_names.*Hopper\)/\1 Blackwell/' cmake/modules/CUDA.cmake

    # Add Blackwell case in architecture selection
    sed -i '/elseif(${CUDA_ARCH_NAME} STREQUAL "Hopper")/,/elseif(${CUDA_ARCH_NAME} STREQUAL "All")/ {
        /set(__cuda_arch_ptx "90")/ a\
  elseif(${CUDA_ARCH_NAME} STREQUAL "Blackwell")\
    set(__cuda_arch_bin "120")\
    set(__cuda_arch_ptx "120")
    }' cmake/modules/CUDA.cmake
fi

echo "Patch applied successfully."

# Create build directory
echo ""
echo "Configuring DGL build with CMake..."
mkdir -p build
cd build

# Configure with CMake
# Use Manual architecture mode and specify sm_120
cmake -DUSE_CUDA=ON \
      -DUSE_OPENMP=ON \
      -DCUDA_ARCH_NAME=Manual \
      -DCUDA_ARCH_BIN="12.0" \
      -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
      ..

# Build C++ library
echo ""
echo "Building DGL C++ library..."
echo "This may take 10-20 minutes..."
make -j$(nproc)

# Install Python package
echo ""
echo "Installing DGL Python package..."
cd ../python
pip install -e .

# Verify installation
echo ""
echo "Verifying DGL installation..."
python -c "import dgl; print(f'DGL Version: {dgl.__version__}')" || {
    echo "Error: DGL installation verification failed."
    exit 1
}

echo ""
echo "================================================================================"
echo "DGL installation completed successfully!"
echo "================================================================================"
echo ""
echo "You can now use DGL with CUDA 12.8 and sm_120 (Blackwell) support."

# Clean up build directory
cd /
echo "Cleaning up build directory: $DGL_BUILD_DIR"
# Uncomment the line below to remove the build directory after installation
# rm -rf $DGL_BUILD_DIR

exit 0
