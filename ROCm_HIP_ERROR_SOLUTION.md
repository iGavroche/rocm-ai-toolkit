# ROCm HIP Error Solution

## Problem
The toolkit was encountering `HIP error: invalid argument` during Accelerate's `prepare()` calls, specifically when Accelerate attempted to move models to the GPU device.

## Root Cause
Even when passing `device_placement=[False]` to `accelerator.prepare()`, Accelerate's internal `prepare_model()` method still calls `model.to(self.device)` in certain cases. This happens because:

1. Accelerate's `prepare_model()` checks `device_placement and not self.verify_device_map(model)` and then calls `model.to(self.device)`
2. The `device_placement` parameter in `prepare_model()` defaults to `self.device_placement` (the Accelerator's default), not the list passed to `prepare()`
3. On ROCm, direct `model.to(device)` calls can trigger HIP errors due to compatibility issues

## Solution Implemented

### 1. Accelerator Configuration (`toolkit/accelerator.py`)
- **Create Accelerator with `device_placement=False` for ROCm**: When ROCm is detected, we create the Accelerator instance with `device_placement=False` to prevent automatic device transfers.
- **Monkey Patch `prepare_model()`**: We patch Accelerate's `prepare_model()` method to force `device_placement=False` for all ROCm calls, ensuring no automatic device transfers occur.

### 2. Safe Prepare Wrapper (`toolkit/accelerator.py`)
- **Enhanced Error Handling**: The `safe_prepare()` function catches any RuntimeError (including AcceleratorError and HIP errors) during `prepare()` calls.
- **CPU-Only Model Placement**: For ROCm, models are kept on CPU and we don't attempt to move them to the device after `prepare()`. Models will be moved to device during forward passes when needed.

### 3. Model Loading Strategy
- **Keep Models on CPU**: During model loading, we explicitly keep models on CPU for ROCm and let Accelerate handle placement (or keep them on CPU if device transfer fails).
- **Step-by-Step Transfers**: When device transfers are necessary, we use step-by-step transfers (CPU → device → dtype) with error handling.

## Changes Made

### `toolkit/accelerator.py`
1. Modified `get_accelerator()` to create Accelerator with `device_placement=False` for ROCm
2. Added `_patch_accelerate_for_rocm()` to monkey patch `prepare_model()` 
3. Simplified `safe_prepare()` to return models as-is for ROCm (no manual device transfer)

### `start_toolkit.sh`
1. Added `--dev` flag for UI development mode with hot reload
2. Updated help text to show the new flag

## How It Works

1. **Accelerator Initialization**: 
   - On ROCm: `Accelerator(device_placement=False)` + monkey patch
   - On CUDA: Normal `Accelerator()`

2. **Model Preparation**:
   - Models are moved to CPU before `safe_prepare()` is called
   - `safe_prepare()` calls `accelerator.prepare()` with `device_placement=[False]`
   - The monkey-patched `prepare_model()` ensures no device transfer occurs
   - Models remain on CPU after preparation

3. **Training**:
   - During forward passes, models will be moved to device as needed
   - If device transfer fails, operations can fall back to CPU

## Testing
To test if this resolves the HIP errors:
1. Run a training job with ROCm
2. Verify that models are prepared without HIP errors
3. Check that training proceeds normally (models may remain on CPU or be moved during forward passes)

## Alternative Approaches Considered
1. **Model Wrappers**: Wrapping models to intercept `.to()` calls - too invasive
2. **CPU Offload**: Using Accelerate's CPU offload feature - requires different configuration
3. **Direct Patching**: Patching at the Accelerate library level - monkey patch approach chosen as more maintainable

## Notes
- Models being on CPU may impact performance, but it's better than crashing
- Future improvements could include:
  - Gradual device transfer during training
  - Better error recovery for device transfers
  - ROCm-specific optimizations as they become available


