#!/usr/bin/env python3
"""
Diagnostic script to test basic GPU operations on ROCm.
This helps identify why GPU operations are failing or causing segfaults.
"""

import sys
import os
import torch

def test_basic_gpu_ops():
    """Test the most basic GPU operations"""
    print("="*60)
    print("Test 1: Basic GPU Detection")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available!")
        return False
    
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ Device count: {torch.cuda.device_count()}")
    print(f"✓ Current device: {torch.cuda.current_device()}")
    print(f"✓ Device name: {torch.cuda.get_device_name(0)}")
    
    # Check ROCm
    try:
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print(f"✓ ROCm version: {torch.version.hip}")
        else:
            print("⚠ Not using ROCm build")
    except:
        pass
    
    print("\n" + "="*60)
    print("Test 2: Simple Tensor Creation on GPU")
    print("="*60)
    
    try:
        device = torch.device("cuda:0")
        print(f"Creating tensor on {device}...")
        x = torch.randn(10, 10, device=device)
        print(f"✓ Tensor created: {x.shape}, device: {x.device}")
        print(f"✓ Tensor operations work: {x.sum().item()}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_torch_randn_gpu():
    """Test torch.randn directly on GPU"""
    print("\n" + "="*60)
    print("Test 3: torch.randn on GPU")
    print("="*60)
    
    try:
        device = torch.device("cuda:0")
        print(f"Calling torch.randn(shape, device={device})...")
        x = torch.randn(100, 100, device=device)
        print(f"✓ torch.randn works: {x.shape}, device: {x.device}")
        return True
    except Exception as e:
        print(f"✗ torch.randn failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_torch_ones_gpu():
    """Test torch.ones directly on GPU"""
    print("\n" + "="*60)
    print("Test 4: torch.ones on GPU")
    print("="*60)
    
    try:
        device = torch.device("cuda:0")
        print(f"Calling torch.ones(shape, device={device})...")
        x = torch.ones(100, 100, device=device)
        print(f"✓ torch.ones works: {x.shape}, device: {x.device}")
        return True
    except Exception as e:
        print(f"✗ torch.ones failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generator_gpu():
    """Test generator on GPU"""
    print("\n" + "="*60)
    print("Test 5: Generator on GPU")
    print("="*60)
    
    try:
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device)
        generator.manual_seed(42)
        print(f"✓ Generator created on {device}")
        
        # Try using generator
        x = torch.randn(10, 10, generator=generator, device=device)
        print(f"✓ Generator works with torch.randn: {x.shape}")
        return True
    except Exception as e:
        print(f"✗ Generator failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_environment():
    """Check environment variables and configuration"""
    print("\n" + "="*60)
    print("Environment Check")
    print("="*60)
    
    rocm_vars = [
        "PYTORCH_ROCM_ARCH",
        "HSA_OVERRIDE_GFX_VERSION",
        "ROCBLAS_USE_HIPBLASLT",
        "PYTORCH_ROCM_ALLOC_CONF",
        "AMD_SERIALIZE_KERNEL",
        "TORCH_USE_HIP_DSA",
        "ROCM_PATH",
        "LD_LIBRARY_PATH",
    ]
    
    for var in rocm_vars:
        value = os.environ.get(var, "not set")
        print(f"{var}: {value}")
    
    print("\nPyTorch Info:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        try:
            if hasattr(torch.version, 'hip'):
                print(f"  HIP version: {torch.version.hip}")
        except:
            pass

def main():
    print("ROCm GPU Basic Operations Diagnostic")
    print("="*60)
    print("This script tests basic GPU operations to identify issues.")
    print("If any test segfaults, there's a fundamental ROCm/PyTorch issue.\n")
    
    check_environment()
    
    results = []
    
    try:
        results.append(("Basic GPU Detection", test_basic_gpu_ops()))
    except Exception as e:
        print(f"\n✗ Basic GPU detection crashed: {e}")
        results.append(("Basic GPU Detection", False))
    
    if results[-1][1]:  # If basic detection worked
        try:
            results.append(("Simple Tensor Creation", test_basic_gpu_ops()))
        except Exception as e:
            print(f"\n✗ Simple tensor creation crashed: {e}")
            results.append(("Simple Tensor Creation", False))
        
        try:
            results.append(("torch.randn on GPU", test_torch_randn_gpu()))
        except Exception as e:
            print(f"\n✗ torch.randn crashed: {e}")
            results.append(("torch.randn on GPU", False))
        
        try:
            results.append(("torch.ones on GPU", test_torch_ones_gpu()))
        except Exception as e:
            print(f"\n✗ torch.ones crashed: {e}")
            results.append(("torch.ones on GPU", False))
        
        try:
            results.append(("Generator on GPU", test_generator_gpu()))
        except Exception as e:
            print(f"\n✗ Generator crashed: {e}")
            results.append(("Generator on GPU", False))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    if not all_passed:
        print("\n" + "="*60)
        print("Diagnosis")
        print("="*60)
        print("GPU operations are failing. Possible causes:")
        print("1. PyTorch ROCm build is not properly configured for gfx1151")
        print("2. ROCm driver/runtime is not properly installed")
        print("3. Memory or driver issue")
        print("4. HIP runtime not properly initialized")
        print("\nRecommendations:")
        print("- Check PyTorch was installed from AMD's gfx1151-specific builds")
        print("- Verify ROCm is properly installed: rocm-smi --version")
        print("- Check dmesg for kernel errors")
        print("- Try reinstalling PyTorch ROCm")
        print("- Consider using CPU fallback for now (code already handles this)")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

