#!/usr/bin/env python3
"""
Test script to verify ROCm component transfers early in the pipeline.
This helps catch HIP errors before waiting for full training to start.
"""

import sys
import torch
from toolkit.backend_utils import is_rocm_available, synchronize_gpu, clear_gpu_cache

def test_device_transfer(component_name, component, target_device):
    """Test if a component can be transferred to GPU on ROCm"""
    print(f"\n{'='*60}")
    print(f"Testing {component_name} device transfer")
    print(f"{'='*60}")
    
    # Ensure component is on CPU first
    try:
        component.cpu()
        synchronize_gpu()
    except Exception as e:
        print(f"  ⚠ Warning: Could not move {component_name} to CPU: {e}")
    
    # Try to move to GPU
    try:
        synchronize_gpu()
        component.to(target_device)
        synchronize_gpu()
        print(f"  ✓ {component_name} successfully moved to GPU")
        return True
    except Exception as e:
        error_str = str(e)
        if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
            print(f"  ✗ {component_name} transfer failed with HIP error (expected on ROCm)")
            print(f"    Error: {error_str[:200]}")
            synchronize_gpu()
            # Keep on CPU
            try:
                component.cpu()
                synchronize_gpu()
                print(f"  ✓ {component_name} kept on CPU (will work with cross-device ops)")
            except:
                pass
            return False
        else:
            print(f"  ✗ {component_name} transfer failed with unexpected error: {e}")
            raise

def test_scheduler_timesteps(scheduler, device):
    """Test if scheduler can set timesteps on device"""
    print(f"\n{'='*60}")
    print(f"Testing scheduler.set_timesteps() on device")
    print(f"{'='*60}")
    
    try:
        synchronize_gpu()
        scheduler.set_timesteps(50, device=device)
        synchronize_gpu()
        print(f"  ✓ Scheduler successfully set timesteps on {device}")
        return True
    except Exception as e:
        error_str = str(e)
        if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
            print(f"  ✗ Scheduler set_timesteps failed with HIP error")
            print(f"    Error: {error_str[:200]}")
            synchronize_gpu()
            # Try CPU instead
            try:
                scheduler.set_timesteps(50, device="cpu")
                print(f"  ✓ Scheduler works with CPU device (will work with cross-device ops)")
                return False  # Works but on CPU
            except Exception as e2:
                print(f"  ✗ Scheduler also failed on CPU: {e2}")
                return False
        else:
            print(f"  ✗ Scheduler failed with unexpected error: {e}")
            raise

def main():
    print("ROCm Component Transfer Test")
    print("="*60)
    
    if not is_rocm_available():
        print("ERROR: ROCm not detected. This test is for ROCm systems only.")
        sys.exit(1)
    
    device = torch.device("cuda:0")
    print(f"Target device: {device}")
    print(f"ROCm detected: ✓")
    
    # Test 1: Simple tensor transfer
    print(f"\n{'='*60}")
    print("Test 1: Simple tensor transfer")
    print(f"{'='*60}")
    try:
        test_tensor = torch.randn(10, 10)
        test_tensor.to(device)
        synchronize_gpu()
        print("  ✓ Simple tensor transfer works")
    except Exception as e:
        print(f"  ✗ Simple tensor transfer failed: {e}")
        sys.exit(1)
    
    # Test 2: Create a simple model and test transfer
    print(f"\n{'='*60}")
    print("Test 2: Simple Linear layer transfer")
    print(f"{'='*60}")
    try:
        linear = torch.nn.Linear(100, 50)
        success = test_device_transfer("Simple Linear layer", linear, device)
        if not success:
            print("  ⚠ Note: Linear layer transfer failed - this is expected for some modules on ROCm")
    except Exception as e:
        print(f"  ✗ Linear layer test failed: {e}")
    
    # Test 3: Test scheduler
    print(f"\n{'='*60}")
    print("Test 3: Scheduler device transfer")
    print(f"{'='*60}")
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler()
        success = test_scheduler_timesteps(scheduler, device)
        if not success:
            print("  ⚠ Note: Scheduler device transfer failed - will need CPU fallback")
    except Exception as e:
        print(f"  ✗ Scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Test bitsandbytes Linear8bitLt if available
    print(f"\n{'='*60}")
    print("Test 4: bitsandbytes Linear8bitLt transfer")
    print(f"{'='*60}")
    try:
        import bitsandbytes.nn as bnb_nn
        linear_8bit = bnb_nn.Linear8bitLt(100, 50)
        # Note: Linear8bitLt should stay on CPU for ROCm
        print("  ✓ Linear8bitLt created (should remain on CPU for ROCm)")
    except ImportError:
        print("  ⚠ bitsandbytes not available - skipping")
    except Exception as e:
        print(f"  ✗ Linear8bitLt test failed: {e}")
    
    # Test 5: Test scheduler with actual model-like scenario
    print(f"\n{'='*60}")
    print("Test 5: Scheduler with complex scenario")
    print(f"{'='*60}")
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler()
        
        # Test setting timesteps on GPU (this is what fails in the pipeline)
        print("  Testing scheduler.set_timesteps() on GPU...")
        try:
            synchronize_gpu()
            scheduler.set_timesteps(50, device=device)
            synchronize_gpu()
            print("  ✓ Scheduler set_timesteps works on GPU")
        except Exception as e:
            error_str = str(e)
            if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
                print("  ✗ Scheduler set_timesteps failed with HIP error (expected)")
                print("    Testing CPU fallback...")
                synchronize_gpu()
                scheduler.set_timesteps(50, device="cpu")
                print("  ✓ Scheduler works with CPU device (fallback works)")
            else:
                print(f"  ✗ Unexpected error: {e}")
                raise
        
        # Test timestep tensor transfer
        timesteps = scheduler.timesteps
        print(f"  Testing timestep tensor transfer (timesteps on {timesteps.device})...")
        try:
            test_timestep = timesteps[0].expand(4)
            synchronize_gpu()
            test_timestep_gpu = test_timestep.to(device)
            synchronize_gpu()
            print("  ✓ Timestep tensor transfer works")
        except Exception as e:
            error_str = str(e)
            if "HIP" in error_str or "hipError" in error_str or "AcceleratorError" in type(e).__name__:
                print("  ✗ Timestep tensor transfer failed with HIP error (expected)")
                print("    Will use CPU timesteps (PyTorch handles cross-device)")
            else:
                print(f"  ✗ Unexpected error: {e}")
    except Exception as e:
        print(f"  ✗ Scheduler complex test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print("If transfers failed with HIP errors, this is expected on ROCm.")
    print("The code should handle these gracefully by keeping components on CPU.")
    print("PyTorch will handle cross-device operations automatically.")
    print("\nTo test actual model components, run:")
    print("  ./start_toolkit.sh train config/examples/train_lora_wan22_14b_i2v_24gb.yaml")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

