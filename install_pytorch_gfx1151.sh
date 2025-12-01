#!/bin/bash
# Install PyTorch with gfx1151 support from AMD's nightly builds
# This fixes "HIP error: invalid device function" on gfx1151 GPUs

set -e

echo "=========================================="
echo "Installing PyTorch with gfx1151 Support"
echo "=========================================="
echo ""

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Activating .venv if it exists..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "✓ Virtual environment activated"
    else
        echo "❌ Error: No virtual environment found"
        echo "   Please create one first: uv venv"
        exit 1
    fi
else
    echo "✓ Virtual environment active: $VIRTUAL_ENV"
fi

echo ""
echo "Current PyTorch installation:"
python3 -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  ROCm: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}')" 2>/dev/null || echo "  PyTorch not installed"

# Detect package manager
if command -v uv &> /dev/null; then
    PKG_MGR="uv pip"
    echo "Using uv as package manager"
elif python3 -m pip --version &> /dev/null; then
    PKG_MGR="python3 -m pip"
    echo "Using pip as package manager"
elif pip3 --version &> /dev/null; then
    PKG_MGR="pip3"
    echo "Using pip3 as package manager"
else
    echo "❌ Error: No package manager found (uv, pip, or pip3)"
    exit 1
fi

echo ""
echo "Uninstalling current PyTorch packages..."
$PKG_MGR uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo ""
echo "Installing PyTorch from AMD's gfx1151-specific nightly builds..."
echo "Index URL: https://rocm.nightlies.amd.com/v2/gfx1151/"
echo ""

if [ "$PKG_MGR" = "uv pip" ]; then
    # Use the exact command the user prefers
    $PKG_MGR install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision --upgrade
else
    # For regular pip, use similar command
    $PKG_MGR install --pre torch torchvision torchaudio --upgrade \
      --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
fi

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if hasattr(torch.version, 'hip') and torch.version.hip:
    print(f'ROCm version: {torch.version.hip}')
else:
    print('ROCm version: N/A')

if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device properties: {torch.cuda.get_device_properties(0)}')
    
    # Test basic GPU operations
    print('')
    print('Testing GPU operations...')
    try:
        x = torch.randn(10, 10).cuda()
        y = torch.matmul(x, x)
        print('✓ Basic GPU operations work')
        
        # Test quantization-like operations
        linear = torch.nn.Linear(10, 10).cuda()
        z = linear(x)
        print('✓ Linear operations work')
        print('✓ GPU quantization should work!')
    except Exception as e:
        print(f'⚠️  GPU test failed: {e}')
        print('   You may still need CPU fallback for some operations')
else:
    print('❌ ROCm not available')
"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Restart your training job"
echo "2. GPU quantization should now work without HIP errors"
echo "3. If issues persist, check GFX1151_HIP_ERROR_FIX.md for troubleshooting"
echo ""

