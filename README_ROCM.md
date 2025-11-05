# ROCm Support for WAN 2.2 LoRA Training

This guide explains how to set up and use ROCm (AMD GPU) support for WAN 2.2 LoRA training, specifically targeting Strix Halo (gfx1151) GPUs.

## Prerequisites

- **Hardware**: AMD GPU with ROCm support, specifically gfx1151 (Strix Halo)
- **Software**: 
  - ROCm 7.1+ (experimental gfx1151 support) or latest nightly builds
  - Python 3.10+ (3.13 tested)
  - `uv` package manager (recommended)

## Installation with uv

### 1. Install uv

If you haven't already, install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via pip:

```bash
pip install uv
```

### 2. Create Virtual Environment

```bash
mkdir ~/wan22_lora
cd ~/wan22_lora
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install ROCm and PyTorch

Install ROCm libraries and PyTorch with ROCm support from AMD's nightly builds:

```bash
# Install ROCm libraries (if needed)
uv pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  "rocm[libraries,devel]"

# Install PyTorch with ROCm support (nightly builds recommended)
uv pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  --pre torch torchaudio torchvision
```

### 4. Set Environment Variables

```bash
export ROCM_PATH=/opt/rocm  # Adjust if ROCm is installed elsewhere
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode
export PYTORCH_ROCM_ARCH="gfx1151"
export ROCBLAS_USE_HIPBLASLT=1  # For optimized performance
```

### 5. Install Project Dependencies

```bash
cd /path/to/ai-toolkit
uv pip install -r requirements.txt
```

Note: PyTorch should already be installed from step 3, so the requirements.txt installation will skip or update it as needed.

## Verification

### Check ROCm Installation

```bash
$ROCM_PATH/bin/rocminfo
# or
$ROCM_PATH/bin/amd-smi
```

### Verify PyTorch ROCm Backend

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Backend: {'ROCm' if hasattr(torch.version, 'hip') and torch.version.hip else 'CUDA'}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

You should see:
- `CUDA available: True` (ROCm PyTorch uses CUDA API)
- `Backend: ROCm`
- Your GPU device name

## Starting the Toolkit

After setup, you can start the toolkit using the provided startup script:

```bash
# Run training with a config file
./start_toolkit.sh train config/examples/train_lora_wan22_14b_24gb.yaml

# Launch Gradio UI
./start_toolkit.sh gradio

# Launch web UI
./start_toolkit.sh ui

# Show help
./start_toolkit.sh help
```

The startup script automatically:
- Detects and configures ROCm/CUDA backend
- Sets up environment variables
- Activates virtual environment if using uv
- Verifies dependencies

## Configuration

When using ROCm, the device string in your config files remains `cuda:0` (ROCm PyTorch uses the CUDA API for compatibility). The code automatically detects the ROCm backend.

Example config (`config/examples/train_lora_wan22_14b_24gb.yaml`):

```yaml
device: cuda:0  # Works for both CUDA and ROCm backends
```

## Known Limitations

1. **Flash Attention**: As of August 2025, PyTorch's Flash Attention may not fully support gfx1151. The code will automatically handle this by using alternative attention backends when needed.

2. **SDP Backends**: Some CUDA-specific Scaled Dot Product (SDP) backends may not be available on ROCm. The code conditionally enables supported backends.

3. **Memory Management**: ROCm uses HIP memory management, which is compatible with PyTorch's CUDA API. The `clear_gpu_cache()` function handles both backends.

## Troubleshooting

### GPU Not Detected

- Ensure ROCm is properly installed: `rocminfo` should show your GPU
- Check environment variables are set correctly
- Verify PyTorch was installed from ROCm nightly builds

### Out of Memory Errors

- Reduce batch size in your training config
- Enable `low_vram: true` in model config
- Use gradient checkpointing: `gradient_checkpointing: true`
- Enable `cache_text_embeddings: true` to reduce memory usage

### Performance Issues

- Ensure `ROCBLAS_USE_HIPBLASLT=1` is set
- Use `bf16` dtype for training (better ROCm performance)
- Consider using ROCm 7.1+ for better gfx1151 support

## Backward Compatibility

The implementation maintains full backward compatibility with CUDA. The code automatically detects the backend (CUDA or ROCm) and adjusts behavior accordingly. All existing configurations work without modification.

## Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Compatibility](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html)
- [AMD TheRock Repository](https://rocm.nightlies.amd.com/)

