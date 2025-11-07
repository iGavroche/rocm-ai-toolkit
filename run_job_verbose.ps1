# PowerShell script to run a training job with verbose output
# Usage: .\run_job_verbose.ps1 config/my_training.yaml
# Or: .\run_job_verbose.ps1 path/to/your/config.yaml

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ConfigFile
)

# Set ROCm environment variables (same as start-ui-strix-halo.ps1)
$env:HSA_XNACK = "0"
$env:HSA_ENABLE_SDMA = "0"
$env:HIP_FORCE_DEV_KERNARG = "1"
$env:HSA_OVERRIDE_GFX_VERSION = "11.5.1"
$env:HIP_MEM_POOL_SUPPORT = "1"
$env:GPU_SINGLE_ALLOC_PERCENT = "100"
$env:HIP_INITIAL_DM_SIZE = "0"
$env:HIP_ENABLE_PEER_ACCESS = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,roundup_power2_divisions:2,max_split_size_mb:256"
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True"
$env:PYTORCH_CUDA_MEMORY_POOL_TYPE = "expandable_segments"
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"
#$env:PYTORCH_NO_HIP_MEMORY_CACHING = "1"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running training job with verbose output" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Config file: $ConfigFile" -ForegroundColor Yellow
Write-Host ""
Write-Host "ROCm environment variables set:" -ForegroundColor Green
Write-Host "  PYTORCH_NO_HIP_MEMORY_CACHING = $env:PYTORCH_NO_HIP_MEMORY_CACHING" -ForegroundColor Green
Write-Host "  HSA_OVERRIDE_GFX_VERSION = $env:HSA_OVERRIDE_GFX_VERSION" -ForegroundColor Green
Write-Host "  PYTORCH_CUDA_ALLOC_CONF = $env:PYTORCH_CUDA_ALLOC_CONF" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    . .venv\Scripts\Activate.ps1
} else {
    Write-Host "Warning: Virtual environment not found at .venv\Scripts\Activate.ps1" -ForegroundColor Red
}

# Run the job with Python's verbose output
Write-Host "Starting job..." -ForegroundColor Yellow
Write-Host ""

python run.py "$ConfigFile" 2>&1 | Tee-Object -FilePath "job_output_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Host ""
Write-Host "Job completed. Check the log file for details." -ForegroundColor Cyan

