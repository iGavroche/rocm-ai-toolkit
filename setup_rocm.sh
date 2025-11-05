#!/bin/bash
# Setup script for ROCm support on Strix Halo (gfx1151)
# This script helps set up the environment with uv for ROCm support

set -e

echo "ROCm Setup Script for WAN 2.2 LoRA Training"
echo "=============================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv is installed"

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.12
    source .venv/bin/activate
    echo "✓ Virtual environment created and activated"
else
    echo "✓ Virtual environment already active: $VIRTUAL_ENV"
fi

# Set environment variables
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode
export PYTORCH_ROCM_ARCH="gfx1151"
export ROCBLAS_USE_HIPBLASLT=1

echo ""
echo "Environment variables set:"
echo "  ROCM_PATH=$ROCM_PATH"
echo "  PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
echo "  ROCBLAS_USE_HIPBLASLT=$ROCBLAS_USE_HIPBLASLT"
echo ""

# Install PyTorch with ROCm support
echo "Installing PyTorch with ROCm support from AMD nightly builds..."
uv pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  --pre torch torchaudio torchvision

echo "✓ PyTorch with ROCm support installed"
echo ""

# Install project dependencies
echo "Installing project dependencies..."
uv pip install -r requirements.txt

echo "✓ Dependencies installed"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if hasattr(torch.version, 'hip') and torch.version.hip:
    print(f'ROCm version: {torch.version.hip}')
    print('✓ ROCm backend detected')
else:
    print('⚠ ROCm backend not detected (may still work if using CUDA API)')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Setup complete!"
echo ""
echo "To use this environment in the future:"
echo "  source .venv/bin/activate"
echo "  export PYTORCH_ROCM_ARCH=\"gfx1151\""
echo "  export ROCBLAS_USE_HIPBLASLT=1"
echo ""
echo "See README_ROCM.md for more information."

