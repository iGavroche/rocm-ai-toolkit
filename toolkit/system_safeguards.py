"""
System safeguards to prevent training from breaking desktop environments.

This module provides safeguards to prevent training from:
- Consuming all system RAM (which could kill GDM/display server)
- Taking exclusive GPU control (which could break display drivers)
- Running on desktop systems without warnings
"""

import os
import sys
import psutil
import resource
from typing import Optional, Tuple


def detect_desktop_environment() -> Tuple[bool, Optional[str]]:
    """
    Detect if running on a desktop environment.
    
    Returns:
        Tuple of (is_desktop, desktop_type)
    """
    # Check for display server
    display = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    if not display:
        return False, None
    
    # Check for common desktop environments
    desktop = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
    if desktop:
        return True, desktop
    
    # Check for X11/Wayland processes
    try:
        for proc in psutil.process_iter(['name']):
            try:
                name = proc.info['name'].lower()
                if any(de in name for de in ['gnome', 'kde', 'xfce', 'mutter', 'kwin', 'gdm', 'lightdm', 'sddm']):
                    return True, 'detected'
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    
    return True, 'unknown'


def check_system_memory(min_free_gb: float = 4.0) -> Tuple[bool, float]:
    """
    Check if system has enough free memory.
    
    Args:
        min_free_gb: Minimum free memory in GB required
        
    Returns:
        Tuple of (has_enough, free_gb)
    """
    mem = psutil.virtual_memory()
    free_gb = mem.available / (1024 ** 3)
    return free_gb >= min_free_gb, free_gb


def set_memory_limit(max_memory_gb: Optional[float] = None) -> None:
    """
    Set soft memory limit for the current process to prevent system RAM exhaustion.
    
    Args:
        max_memory_gb: Maximum memory in GB (None = 80% of available RAM)
    
    Note: Memory limits can cause issues with PyTorch/GPU training. 
    This is disabled by default - enable via ENABLE_MEMORY_LIMIT=1 environment variable.
    """
    # Check if memory limits are enabled (disabled by default due to PyTorch compatibility issues)
    if os.environ.get('ENABLE_MEMORY_LIMIT') != '1':
        print("[System Safeguard] Memory limits disabled by default (can cause issues with PyTorch/GPU training)")
        print("[System Safeguard] To enable, set ENABLE_MEMORY_LIMIT=1 environment variable")
        return
    
    try:
        # Check if resource module supports RLIMIT_AS (Unix/Linux only)
        if not hasattr(resource, 'RLIMIT_AS'):
            print("[System Safeguard] Memory limits not supported on this platform (Windows?)")
            return
        
        # Get available memory
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        
        # Default to 80% of available RAM, but leave at least 4GB free
        if max_memory_gb is None:
            max_memory_gb = max(available_gb * 0.8, available_gb - 4.0)
        
        # Set soft limit (process will get SIGXCPU/SIGXFSZ warnings, but won't be killed)
        # Convert GB to bytes
        max_bytes = int(max_memory_gb * 1024 ** 3)
        
        # Set RLIMIT_AS (address space limit)
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
        
        # Handle RLIMIT_INFINITY - it might be a constant or -1 depending on platform
        if hasattr(resource, 'RLIMIT_INFINITY'):
            rlimit_infinity = resource.RLIMIT_INFINITY
        else:
            # On some systems, unlimited is represented as -1
            rlimit_infinity = -1
        
        # Check if hard limit is unlimited
        if hard_limit == rlimit_infinity or hard_limit == -1:
            new_soft = max_bytes
        else:
            new_soft = min(max_bytes, hard_limit)
        
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard_limit))
        
        print(f"[System Safeguard] Set memory limit to {max_memory_gb:.1f}GB ({max_bytes / (1024**3):.1f}GB)")
    except (AttributeError, OSError, ValueError) as e:
        print(f"[System Safeguard] Warning: Could not set memory limit: {e}")
        print("[System Safeguard] Memory limits may not be supported on this platform")
    except Exception as e:
        print(f"[System Safeguard] Warning: Could not set memory limit: {e}")


def check_gpu_sharing() -> bool:
    """
    Check if GPU is being shared with display server.
    
    Returns:
        True if GPU sharing is detected
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        
        # Try to create a small tensor - if this fails, GPU might be exclusively locked
        try:
            test_tensor = torch.zeros(1, device='cuda:0')
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except Exception:
            return False
    except Exception:
        return False


def is_interactive() -> bool:
    """
    Check if running in an interactive environment (can get user input).
    
    Returns:
        True if interactive, False otherwise
    """
    # Check if stdin is a TTY (terminal)
    if hasattr(sys.stdin, 'isatty'):
        if not sys.stdin.isatty():
            return False
    
    # Check for UI environment variable (set by UI when starting jobs)
    if os.environ.get('IS_AI_TOOLKIT_UI') == '1':
        return False
    
    # Check for skip safeguards environment variable
    if os.environ.get('SKIP_SAFEGUARDS_CONFIRMATION') == '1':
        return False
    
    return True


def warn_desktop_environment() -> bool:
    """
    Warn user if running on desktop environment and ask for confirmation.
    
    Returns:
        True if user confirms or non-interactive, False otherwise
    """
    is_desktop, desktop_type = detect_desktop_environment()
    
    if not is_desktop:
        return True
    
    print("\n" + "=" * 70)
    print("⚠️  WARNING: DESKTOP ENVIRONMENT DETECTED")
    print("=" * 70)
    print(f"Desktop type: {desktop_type}")
    print("\nRunning training on a desktop system can be DANGEROUS:")
    print("  • Training may consume all system RAM, causing GDM/display server to crash")
    print("  • GPU resource conflicts may break display drivers")
    print("  • System may become unusable or require recovery")
    print("\nRECOMMENDATIONS:")
    print("  • Use a dedicated GPU for training (if available)")
    print("  • Set lower batch sizes and model offloading")
    print("  • Monitor system resources during training")
    print("  • Consider using a headless server or virtual machine")
    print("\n" + "=" * 70)
    
    # Check system memory
    has_enough, free_gb = check_system_memory()
    if not has_enough:
        print(f"\n⚠️  WARNING: Low system memory ({free_gb:.1f}GB free)")
        print("   Risk of OOM killer terminating system processes is HIGH")
        print("   Training may break your desktop environment!")
        # In non-interactive mode, still proceed but with warnings
        if not is_interactive():
            print("\n⚠️  Running in non-interactive mode - proceeding with safeguards enabled")
            print("   Memory limits will be set to protect system processes")
            return True
    else:
        print(f"\n✓ System memory OK ({free_gb:.1f}GB free)")
    
    # If not interactive (UI or automated), skip confirmation but show warning
    if not is_interactive():
        print("\n⚠️  Running in non-interactive mode (UI/automated)")
        print("   Proceeding with safeguards enabled (memory limits will be set)")
        print("   To disable safeguards, set SKIP_SAFEGUARDS=1 environment variable")
        return True
    
    # Ask for confirmation in interactive mode
    print("\nDo you want to continue? (yes/no): ", end='', flush=True)
    
    try:
        response = input().strip().lower()
        if response in ['yes', 'y']:
            print("Proceeding with training...")
            return True
        else:
            print("Training cancelled.")
            return False
    except (EOFError, KeyboardInterrupt):
        print("\nTraining cancelled.")
        return False


def setup_system_safeguards(require_confirmation: bool = False) -> bool:
    """
    Set up system safeguards before training.
    
    Args:
        require_confirmation: Whether to require user confirmation on desktop systems
                             (automatically disabled for non-interactive environments)
        
    Returns:
        True if safeguards are OK and training can proceed
    """
    # Check if safeguards are disabled via environment variable
    if os.environ.get('SKIP_SAFEGUARDS') == '1':
        print("[System Safeguard] ⚠️  WARNING: Safeguards disabled via SKIP_SAFEGUARDS=1")
        print("[System Safeguard] Proceeding without safeguards (RISKY)")
        return True
    
    # Detect desktop environment
    is_desktop, desktop_type = detect_desktop_environment()
    
    # Auto-disable confirmation for non-interactive environments (UI, automated)
    if not is_interactive():
        require_confirmation = False
        if is_desktop:
            print(f"[System Safeguard] Desktop environment detected: {desktop_type}")
            print("[System Safeguard] Running in non-interactive mode - safeguards enabled automatically")
    
    if is_desktop:
        print(f"[System Safeguard] Desktop environment detected: {desktop_type}")
        
        # Check system memory
        has_enough, free_gb = check_system_memory()
        if not has_enough:
            print(f"[System Safeguard] ⚠️  WARNING: Low system memory ({free_gb:.1f}GB free)")
            print("[System Safeguard] Risk of OOM killer terminating system processes")
            
            # Critical check: if memory is critically low (<2GB), fail even in non-interactive mode
            if free_gb < 2.0:
                print(f"[System Safeguard] ❌ ERROR: Critically low memory ({free_gb:.1f}GB free)")
                print("[System Safeguard] Cannot safely run training - risk of breaking desktop environment")
                print("[System Safeguard] Please free up memory or reduce training configuration")
                return False
        
        # Skip confirmation - auto-continue (user requested)
        # Only show warning, don't ask for confirmation
        if require_confirmation and is_interactive():
            if not warn_desktop_environment():
                return False
        else:
            # Auto-continue with just a warning
            print(f"[System Safeguard] Proceeding with training on desktop system (auto-continue)")
        
        # Set memory limits to prevent system RAM exhaustion
        # Leave at least 4GB for system processes
        set_memory_limit()
    else:
        print("[System Safeguard] No desktop environment detected (headless/server mode)")
    
    # Check GPU sharing
    if not check_gpu_sharing():
        print("[System Safeguard] ⚠️  WARNING: GPU may not be accessible (check display server)")
    
    return True

