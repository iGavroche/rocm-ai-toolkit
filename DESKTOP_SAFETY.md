# Desktop Environment Safety Guide

## ⚠️ IMPORTANT WARNING

**Running training on a desktop system can BREAK your desktop environment (GNOME, KDE, etc.)**

This has happened to users. Training can:
- Consume all system RAM, causing the OOM killer to terminate GDM/display server
- Cause GPU driver conflicts between ROCm/CUDA and display drivers
- Make your system unusable, requiring recovery from a live disk

## What Happened

In one case, training on a ROCm-enabled system (Strix Halo) caused:
- Complete failure of GNOME/GDM
- System became unusable
- Required reinstallation of GNOME/GDM from a live disk

**Possible causes:**
1. **System RAM exhaustion**: Training consumed all available RAM, OOM killer terminated critical system processes (GDM)
2. **GPU driver conflicts**: ROCm took exclusive control of GPU, preventing display server from accessing it
3. **Resource exhaustion**: Training used all system resources without limits

## Safeguards Added

The toolkit now includes automatic safeguards that:

1. **Detect desktop environments** - Checks if running on GNOME, KDE, X11, Wayland, etc.
2. **Check system memory** - Warns if less than 4GB free RAM available
3. **Set memory limits** - Limits training process to 80% of available RAM (leaves 4GB+ for system)
4. **Require confirmation** - Asks user to confirm before training on desktop systems
5. **GPU sharing check** - Verifies GPU is accessible (not exclusively locked)

## How to Use Safeguards

### Default Behavior (Recommended)

When you run training, the safeguards will:
1. Detect if you're on a desktop system
2. Check available memory
3. **Ask for confirmation** before proceeding
4. Set memory limits to protect system processes

```bash
./start_toolkit.sh train config/examples/train_lora_wan22_14b_24gb.yaml
```

You'll see a warning like:
```
⚠️  WARNING: DESKTOP ENVIRONMENT DETECTED
Desktop type: gnome

Running training on a desktop system can be DANGEROUS:
  • Training may consume all system RAM, causing GDM/display server to crash
  • GPU resource conflicts may break display drivers
  • System may become unusable or require recovery

Do you want to continue? (yes/no):
```

### Skip Safeguards (NOT RECOMMENDED)

Only skip safeguards if:
- You're running on a headless server
- You have a dedicated GPU for training (separate from display GPU)
- You understand the risks and have backups

```bash
python run.py config/examples/train_lora_wan22_14b_24gb.yaml --skip-safeguards
```

## Recommendations

### For Desktop Systems

1. **Use a dedicated GPU** - If you have multiple GPUs, use one for training and one for display
2. **Lower batch sizes** - Reduce memory usage with smaller batches
3. **Enable model offloading** - Use `low_vram: true` in config to offload models to CPU
4. **Monitor resources** - Watch system memory and GPU usage during training
5. **Use headless mode** - Consider running training on a server or VM instead

### Example Safe Config for Desktop

```yaml
device: cuda:0
model:
  low_vram: true  # Offload models to CPU when not in use
train:
  batch_size: 1   # Lower batch size to reduce memory
  gradient_accumulation_steps: 4  # Simulate larger batch size
```

### For Headless Servers

On headless servers (no desktop environment), safeguards will:
- Detect no desktop environment
- Skip confirmation prompt
- Still set memory limits (but less restrictive)

## Technical Details

### Memory Limits

The safeguards use `resource.setrlimit()` to set a soft memory limit:
- **Desktop systems**: Limits to 80% of available RAM, minimum 4GB free
- **Headless systems**: Limits to 80% of available RAM

This prevents the OOM killer from terminating system processes, but training may still fail with memory errors (which is safer than killing GDM).

### Detection Methods

Desktop detection checks:
1. `DISPLAY` or `WAYLAND_DISPLAY` environment variables
2. `XDG_CURRENT_DESKTOP` environment variable
3. Running processes (gdm, gnome-shell, mutter, kwin, etc.)

### GPU Sharing

Checks if GPU is accessible by creating a test tensor. If this fails, GPU might be exclusively locked by another process.

## If Training Breaks Your System

1. **Reboot into recovery mode** or use a live disk
2. **Check system logs**: `journalctl -b -1` (previous boot)
3. **Check OOM killer logs**: `dmesg | grep -i oom`
4. **Reinstall display manager**: 
   ```bash
   # For GNOME/GDM
   sudo pacman -S gdm gnome-session gnome-shell
   
   # For KDE/SDDM
   sudo pacman -S sddm plasma kde-applications
   ```
5. **Consider running training on a server/VM** instead

## Reporting Issues

If you experience system issues related to training:
1. Check system logs (`journalctl`, `dmesg`)
2. Check available memory before/after training
3. Note your GPU configuration (ROCm/CUDA, single/multiple GPUs)
4. Report with full system details

## Additional Resources

- [ROCm Configuration Guide](ROCm_CONFIGURATION.md)
- [System Resource Management](https://www.kernel.org/doc/html/latest/admin-guide/mm/oom_killer.html)


