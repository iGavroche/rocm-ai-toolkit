# ROCm GPU Quantization Failures - Complete History

## Summary
All attempts to enable GPU quantization on ROCm (gfx1151/Strix Halo) have failed. The root cause is that PyTorch ROCm builds lack proper kernel support for gfx1151 architecture, causing HIP errors even for small module transfers.

## Failure Timeline

### Attempt 1: Direct GPU Transfer
- **Approach**: Transfer entire model to GPU, then quantize
- **Result**: Failed with "HIP error: invalid argument"
- **Root Cause**: PyTorch ROCm doesn't support large model transfers for gfx1151

### Attempt 2: Chunked Parameter Transfer
- **Approach**: Transfer parameters one-by-one to GPU
- **Result**: Failed - even individual parameters cause HIP errors
- **Root Cause**: Same kernel support issue

### Attempt 3: Module-by-Module GPU Transfer
- **Approach**: Move modules one at a time to GPU, quantize immediately
- **Result**: **FAILED** - 80+ modules out of 406 failed with HIP errors
- **Details**:
  - All failed modules fell back to CPU quantization
  - Process killed with OOM after 82/406 modules
  - CPU RAM exhausted (not using VRAM)
  - Speed degraded to 11.52s/it before OOM kill
- **Root Cause**: Even small individual modules (Linear, Conv2d) fail to transfer to GPU

### Attempt 4: CPU Quantization with Memory Management
- **Approach**: Quantize on CPU with aggressive cleanup
- **Result**: Still causes OOM - memory cleanup doesn't prevent accumulation
- **Root Cause**: Large models (WAN 2.2 14B) require too much RAM for CPU quantization

## Root Cause Analysis

1. **PyTorch ROCm Build Issue**: 
   - Version: `2.10.0a0+rocm7.10.0a20251015`
   - Lacks proper gfx1151 kernel support for device transfers
   - Even small modules fail with HIP errors

2. **Architecture Limitation**:
   - gfx1151 (Strix Halo) is relatively new
   - ROCm support is still maturing
   - PyTorch ROCm builds may not have full kernel coverage

3. **Memory Constraints**:
   - CPU quantization of 14B models requires ~30GB+ RAM
   - System has limited RAM (gets exhausted)
   - VRAM (96GB) is available but can't be used due to transfer failures

## Current Status

**GPU Quantization**: ❌ **NOT POSSIBLE** with current PyTorch ROCm build
- All transfer attempts fail with HIP errors
- Even individual modules fail
- No workaround found

**CPU Quantization**: ⚠️ **WORKS BUT CAUSES OOM**
- Quantization succeeds but uses system RAM
- Large models (14B) exhaust available RAM
- Memory cleanup doesn't prevent accumulation

## Solutions

### Short-term (Workarounds)
1. **Increase Swap Space**: Add more swap to handle RAM overflow
2. **Use Pre-quantized Models**: Quantize offline on machine with more RAM
3. **Disable Quantization**: Set `quantize: false` in config (uses more VRAM but avoids OOM)
4. **Quantize Only Specific Layers**: Use `include` patterns to quantize only transformer blocks

### Long-term (Proper Fix)
1. **Wait for Better PyTorch ROCm Build**: 
   - Need PyTorch build with proper gfx1151 kernel support
   - Monitor AMD's gfx1151-specific nightly builds
   - Check: https://rocm.nightlies.amd.com/v2/gfx1151/

2. **Use Alternative Quantization Libraries**:
   - **bitsandbytes ROCm**: Build from source for gfx1151
   - **AMD Quark**: AMD's official quantization library
   - **vLLM with PTPC-FP8**: For inference (not training)

3. **Upgrade PyTorch**:
   - Try newer PyTorch ROCm builds
   - Check for gfx1151-specific builds
   - May require building from source

## Recommendations

**For Now**: 
- Disable quantization (`quantize: false`) to avoid OOM
- Model will use more VRAM but should fit in 96GB
- Training will work, just slower without quantization

**For Future**:
- Monitor PyTorch ROCm releases for gfx1151 support
- Consider using pre-quantized model checkpoints
- Explore alternative quantization libraries (bitsandbytes ROCm, AMD Quark)

## Files Modified
- `toolkit/util/quantize.py` - All quantization logic attempts
- `start_toolkit.sh` - ROCm environment variables
- `ROCM_QUANTIZATION_SETUP.md` - Setup documentation
- `ROCM_QUANTIZATION_ALTERNATIVES.md` - Alternative approaches

## Test Results
- GPU transfer: ❌ Fails for entire model
- GPU transfer: ❌ Fails for individual modules (80+ failures)
- CPU quantization: ⚠️ Works but causes OOM
- Memory cleanup: ❌ Doesn't prevent RAM exhaustion


