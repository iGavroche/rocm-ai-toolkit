# ROCm GPU Placement Solution (Matching musubi-tuner)

## Problem
The toolkit was moving models to CPU before `prepare_accelerator()` and keeping them there for ROCm, causing CPU/RAM usage and swapping. This was inefficient and didn't match the musubi-tuner approach.

## Root Cause
The previous implementation:
1. Moved models to CPU before `prepare_accelerator()` 
2. Used `device_placement=False` to prevent Accelerate from placing models on GPU
3. Kept models on CPU after prepare, causing CPU/RAM usage and swapping

## Solution Implemented (Matching musubi-tuner)

### 1. Accelerator Configuration (`toolkit/accelerator.py`)
- **Create Accelerator with `device_placement=True` for ROCm**: When ROCm is detected, we create the Accelerator instance with `device_placement=True` (default) to allow GPU placement.
- **Monkey Patch `prepare_model()`**: We patch Accelerate's `prepare_model()` method to allow device placement (defaults to True) with error handling for HIP errors.

### 2. Safe Prepare Wrapper (`toolkit/accelerator.py`)
- **Allow GPU Placement**: The `safe_prepare()` function uses `device_placement=[True]` for ROCm to allow Accelerate to place models on GPU.
- **No CPU Forcing**: Models are NOT moved to CPU before prepare - they should already be on GPU from model loading.
- **Error Handling**: HIP errors during prepare are caught and logged, but models stay on GPU.

### 3. Model Loading Strategy
- **Load Models to GPU**: During model loading (in `wan21.py`, `wan22_14b_model.py`), models are loaded directly to `accelerator.device` (GPU).
- **Keep Models on GPU**: Models remain on GPU throughout the training process, matching musubi-tuner's approach.
- **Explicit Device Placement**: After `prepare_accelerator()`, models are explicitly moved to `accelerator.device` to ensure they're on GPU.

## Changes Made

### `toolkit/accelerator.py`
1. Modified `get_accelerator()` to create Accelerator with `device_placement=True` for ROCm (matching musubi-tuner)
2. Updated `_patch_accelerate_for_rocm()` to allow device placement (defaults to True) with error handling
3. Modified `safe_prepare()` to use `device_placement=[True]` and NOT move models to CPU

### `jobs/process/BaseSDTrainProcess.py`
1. Removed all CPU movement before `prepare_accelerator()`
2. Changed `safe_prepare()` calls to use `device_placement=[True]`
3. Added explicit device placement after prepare to ensure models are on `accelerator.device`

## How It Works (Matching musubi-tuner)

1. **Accelerator Initialization**: 
   - On ROCm: `Accelerator(device_placement=True)` + monkey patch (allows GPU placement)
   - On CUDA: Normal `Accelerator()`

2. **Model Loading**:
   - Models are loaded directly to `accelerator.device` (GPU) during model loading
   - This happens in `wan21.py` and `wan22_14b_model.py` - models are moved to GPU after loading

3. **Model Preparation**:
   - Models are already on GPU from loading (NOT moved to CPU)
   - `safe_prepare()` calls `accelerator.prepare()` with `device_placement=[True]`
   - The monkey-patched `prepare_model()` allows device placement (defaults to True)
   - Models remain on GPU after preparation
   - Explicit `.to(accelerator.device)` calls ensure models are on GPU

4. **Training**:
   - Models stay on GPU throughout training
   - No CPU/RAM usage or swapping
   - LoRA modules remain on same device as parent modules

## Testing
To test if this works correctly:
1. Run a training job with ROCm
2. Verify that models stay on GPU (check `rocm-smi` or `nvidia-smi`)
3. Check that there's no CPU/RAM spiking or swapping
4. Verify training proceeds normally without HIP errors
5. Confirm LoRA modules are on the same device as parent modules

## Key Differences from Previous Approach
- **Previous**: Models moved to CPU → kept on CPU → caused RAM/swap usage
- **Current**: Models loaded to GPU → kept on GPU → no RAM/swap usage (matching musubi-tuner)

## Notes
- This matches musubi-tuner's approach exactly: load to `accelerator.device` and keep there
- Models being on GPU is more efficient and prevents CPU/RAM issues
- HIP errors are handled gracefully but don't force CPU fallback
  - Better error recovery for device transfers
  - ROCm-specific optimizations as they become available


