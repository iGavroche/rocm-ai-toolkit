# AI Toolkit UI Startup Script with Strix Halo (gfx1151) ROCm Configuration
# This script sets environment variables to force dedicated GPU memory usage
# and starts the AI Toolkit UI service

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AI Toolkit UI - Strix Halo Configuration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the project root
if (-not (Test-Path "ui\package.json")) {
    Write-Host "ERROR: This script must be run from the project root directory!" -ForegroundColor Red
    Write-Host "Please navigate to the ai-toolkit root directory and run this script again." -ForegroundColor Red
    exit 1
}

Write-Host "Setting ROCm/HIP environment variables for Strix Halo (gfx1151)..." -ForegroundColor Yellow
Write-Host ""

# # CRITICAL: Disable unified memory/page migration on Strix Halo
# # HSA_XNACK=0 prevents using system RAM as GPU memory
# $env:HSA_XNACK = "0"

# # Disable SDMA (System DMA) to prevent shared memory transfers
# $env:HSA_ENABLE_SDMA = "0"

# # Force device kernel argument allocation (prevents using system memory for kernel args)
# $env:HIP_FORCE_DEV_KERNARG = "1"

# # Override GFX version for gfx1151 (Strix Halo)
# $env:HSA_OVERRIDE_GFX_VERSION = "11.5.1"

# # Force memory pool to use only device memory, not system memory
# $env:HIP_MEM_POOL_SUPPORT = "1"

# # Limit allocation to device memory only (prevent fallback to shared memory)
# $env:GPU_SINGLE_ALLOC_PERCENT = "100"

# # Initialize device memory pool immediately
# $env:HIP_INITIAL_DM_SIZE = "0"  # 0 = use all available device memory

# # Disable peer-to-peer memory access (which can trigger shared memory)
# $env:HIP_ENABLE_PEER_ACCESS = "0"

# # PyTorch memory allocator configuration (optimized for ROCm/gfx1151 based on rocm-ninodes)
# # Reference: https://github.com/iGavroche/rocm-ninodes
# # expandable_segments helps with memory fragmentation on newer architectures like gfx1151
# # roundup_power2_divisions helps with memory alignment
# # max_split_size_mb:512 is recommended by rocm-ninodes for gfx1151
# $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,roundup_power2_divisions:2,max_split_size_mb:512"

# # ROCm-specific HIP allocator configuration (rocm-ninodes recommendation)
# $env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True"

# # Memory pool type configuration (rocm-ninodes recommendation)
# $env:PYTORCH_CUDA_MEMORY_POOL_TYPE = "expandable_segments"

# # Optional: Set CUDA/HIP device order
# $env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"

# # Optional: Disable diffusers telemetry
# $env:DISABLE_TELEMETRY = "YES"

# # Optional: Enable HF transfer for faster downloads
# $env:HF_HUB_ENABLE_HF_TRANSFER = "1"

# # Optional: Disable albumentations update check
# $env:NO_ALBUMENTATIONS_UPDATE = "1"

Set-Location "C:\Users\Nino\projects\ai-toolkit"
& ".\.venv\Scripts\Activate"

# Uncomment to reinstall PyTorch with ROCm nightlies (only if needed)
# uv pip uninstall torch torchaudio torchvision
# uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision --upgrade

# PyTorch memory allocator configuration
# Allow override via PYTORCH_MAX_SPLIT_SIZE_MB environment variable
# Default: 256 (reduced from 512 for severe fragmentation issues)
# For even more severe fragmentation, try: 128 or 64
$maxSplit = if ($env:PYTORCH_MAX_SPLIT_SIZE_MB) { $env:PYTORCH_MAX_SPLIT_SIZE_MB } else { "768" }
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,roundup_power2_divisions:2,max_split_size_mb:$maxSplit"
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True"
$env:PYTORCH_CUDA_MEMORY_POOL_TYPE = "expandable_segments"

# CRITICAL FIX for unified memory architecture (gfx1151/Strix Halo):
# Disable PyTorch's caching allocator which doesn't properly account for unified memory
# The caching allocator reserves memory blocks that it thinks are free but aren't actually
# available due to unified memory architecture confusion
# This forces PyTorch to allocate/deallocate memory directly without caching
# 
# WARNING: Setting this to "1" causes ROCm driver crashes (0xC0000005) with quantized models
# If you experience crashes, leave this commented out (enables caching, may cause fragmentation)
# $env:PYTORCH_NO_HIP_MEMORY_CACHING = "1"

# Experimental ROCm features
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"

# GFX version override - try 11.0.0 if 11.5.1 doesn't work
# Uncomment and set to "11.0.0" if experiencing runtime detection issues
# $env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"

Write-Host "Environment variables set:" -ForegroundColor Green
Write-Host "  PYTORCH_CUDA_ALLOC_CONF = $env:PYTORCH_CUDA_ALLOC_CONF" -ForegroundColor Gray
Write-Host "  PYTORCH_HIP_ALLOC_CONF = $env:PYTORCH_HIP_ALLOC_CONF" -ForegroundColor Gray
Write-Host "  PYTORCH_CUDA_MEMORY_POOL_TYPE = $env:PYTORCH_CUDA_MEMORY_POOL_TYPE" -ForegroundColor Gray
Write-Host "  PYTORCH_NO_HIP_MEMORY_CACHING = $env:PYTORCH_NO_HIP_MEMORY_CACHING" -ForegroundColor Yellow
Write-Host "  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" -ForegroundColor Gray
if ($env:HSA_OVERRIDE_GFX_VERSION) {
    Write-Host "  HSA_OVERRIDE_GFX_VERSION = $env:HSA_OVERRIDE_GFX_VERSION" -ForegroundColor Gray
}
Write-Host ""
Write-Host "CRITICAL FIX APPLIED:" -ForegroundColor Yellow
Write-Host "  PYTORCH_NO_HIP_MEMORY_CACHING=1 disables PyTorch caching allocator" -ForegroundColor Gray
Write-Host "  This fixes unified memory architecture confusion on gfx1151/Strix Halo" -ForegroundColor Gray
Write-Host ""
Write-Host "Troubleshooting tips:" -ForegroundColor Yellow
Write-Host "  - If experiencing fragmentation, set: `$env:PYTORCH_MAX_SPLIT_SIZE_MB = '256'` or `'128'`" -ForegroundColor Gray
Write-Host "  - If runtime detection fails, try: `$env:HSA_OVERRIDE_GFX_VERSION = '11.0.0'`" -ForegroundColor Gray
Write-Host "  - fp32 is preferred over bf16 for best ROCm performance on gfx1151" -ForegroundColor Gray
Write-Host "  - See ROCm_gfx1151_troubleshooting.md for more details" -ForegroundColor Gray
Write-Host ""

# Check if AI_TOOLKIT_AUTH is set
if ($env:AI_TOOLKIT_AUTH) {
    Write-Host "AI_TOOLKIT_AUTH is set (UI will be password protected)" -ForegroundColor Yellow
} else {
    Write-Host "AI_TOOLKIT_AUTH is not set (UI will be unprotected)" -ForegroundColor Yellow
    Write-Host "  Set it with: `$env:AI_TOOLKIT_AUTH = 'your_password'" -ForegroundColor Gray
}
Write-Host ""

Write-Host "Starting UI service in DEV mode..." -ForegroundColor Yellow
Write-Host "  Navigate to http://localhost:8675 after startup" -ForegroundColor Gray
Write-Host "  Changes will hot-reload automatically (no rebuild needed)" -ForegroundColor Gray
Write-Host ""

# Change to ui directory and start the service
Set-Location ui

# Check if node_modules exists, if not run install first
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies (first time only)..." -ForegroundColor Yellow
    npm install
    Write-Host ""
}

# Check if database needs to be updated
Write-Host "Updating database schema..." -ForegroundColor Yellow
npm run update_db
Write-Host ""

try {
    # Run dev mode which uses hot-reload (no rebuild needed)
    npm run dev
} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to start UI service!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Set-Location ..
    exit 1
}

Set-Location "C:\Users\Nino\projects\ai-toolkit"



