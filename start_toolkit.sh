#!/bin/bash
# Startup script for AI Toolkit with ROCm/CUDA support
# This script sets up the environment and launches the toolkit

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="help"
CONFIG_FILE=""
RECOVER=false
JOB_NAME=""
LOG_FILE=""
UI_PORT=8675

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "AI Toolkit Startup Script"
    echo "=========================="
    echo ""
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  train <config_file> [config_file2 ...]  - Run training job(s) with config file(s)"
    echo "  gradio                                  - Launch Gradio UI for FLUX training"
    echo "  ui                                       - Launch web UI (Next.js, production mode)"
    echo "  ui --dev                                 - Launch web UI (development mode with hot reload)"
    echo "  help                                     - Show this help message"
    echo ""
    echo "Training Options:"
    echo "  -r, --recover                            - Continue running additional jobs if one fails"
    echo "  -n, --name NAME                          - Name to replace [name] tag in config"
    echo "  -l, --log FILE                           - Log file to write output to"
    echo ""
    echo "UI Options:"
    echo "  -p, --port PORT                          - Port for web UI (default: 8675)"
    echo ""
    echo "Examples:"
    echo "  $0 train config/examples/train_lora_wan22_14b_24gb.yaml"
    echo "  $0 train config/my_config.yaml -r -n my_training_run"
    echo "  $0 gradio"
    echo "  $0 ui"
    echo ""
}

# Function to detect backend
detect_backend() {
    print_info "Detecting GPU backend..."
    
    if python3 -c "import torch; print('ROCm' if hasattr(torch.version, 'hip') and torch.version.hip else 'CUDA')" 2>/dev/null; then
        BACKEND=$(python3 -c "import torch; print('ROCm' if hasattr(torch.version, 'hip') and torch.version.hip else 'CUDA')" 2>/dev/null)
        print_success "Backend detected: $BACKEND"
        
        if [ "$BACKEND" = "ROCm" ]; then
            # Set ROCm environment variables if not already set
            if [ -z "$ROCM_PATH" ]; then
                if [ -d "/opt/rocm" ]; then
                    export ROCM_PATH=/opt/rocm
                else
                    print_warning "ROCM_PATH not set and /opt/rocm not found. ROCm may not work correctly."
                fi
            fi
            
            if [ -z "$PYTORCH_ROCM_ARCH" ]; then
                export PYTORCH_ROCM_ARCH="gfx1151"
                print_info "Set PYTORCH_ROCM_ARCH=gfx1151"
            fi
            
            if [ -z "$ROCBLAS_USE_HIPBLASLT" ]; then
                export ROCBLAS_USE_HIPBLASLT=1
                print_info "Set ROCBLAS_USE_HIPBLASLT=1"
            fi
            
            # Set AMD_SERIALIZE_KERNEL for better error reporting (as suggested by HIP errors)
            if [ -z "$AMD_SERIALIZE_KERNEL" ]; then
                export AMD_SERIALIZE_KERNEL=3
                print_info "Set AMD_SERIALIZE_KERNEL=3 for better error reporting"
            fi
            
            # Set TORCH_USE_HIP_DSA to enable device-side assertions (as suggested by HIP errors)
            if [ -z "$TORCH_USE_HIP_DSA" ]; then
                export TORCH_USE_HIP_DSA=1
                print_info "Set TORCH_USE_HIP_DSA=1 for device-side assertions"
            fi
            
            # Set HSA_OVERRIDE_GFX_VERSION for Strix Halo (gfx1151) compatibility
            if [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
                export HSA_OVERRIDE_GFX_VERSION=11.0.0
                print_info "Set HSA_OVERRIDE_GFX_VERSION=11.0.0 for gfx1151 compatibility"
            fi
            
            # Set HIP_LAUNCH_BLOCKING for debugging (optional, can be disabled for performance)
            # Setting to 1 makes kernel launches synchronous for better error reporting
            if [ -z "$HIP_LAUNCH_BLOCKING" ]; then
                # Default to 0 for performance, but can be set to 1 for debugging
                export HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-0}"
                if [ "$HIP_LAUNCH_BLOCKING" = "1" ]; then
                    print_info "HIP_LAUNCH_BLOCKING=1 (synchronous kernels for debugging)"
                fi
            fi
            
            # Additional ROCm tuning for APU/quantization (from Grok's analysis)
            # Disable SDMA if conflicting on APUs (common for memcpy in quant)
            if [ -z "$HSA_ENABLE_SDMA" ]; then
                export HSA_ENABLE_SDMA=0
                print_info "Set HSA_ENABLE_SDMA=0 (disable SDMA for APU compatibility)"
            fi
            
            # Better VRAM fragmentation for large shared memory pools (128GB EVO-X2)
            if [ -z "$PYTORCH_ROCM_ALLOC_CONF" ]; then
                export PYTORCH_ROCM_ALLOC_CONF="max_split_size_mb:256,garbage_collect=1"
                print_info "Set PYTORCH_ROCM_ALLOC_CONF for better VRAM fragmentation"
            fi
            
            # Additional ROCm optimization variables
            # HIP_VISIBLE_DEVICES can be used to select specific GPUs (similar to CUDA_VISIBLE_DEVICES)
            if [ -n "$HIP_VISIBLE_DEVICES" ]; then
                print_info "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
            fi
            
            # Set library paths
            if [ -n "$ROCM_PATH" ]; then
                export LD_LIBRARY_PATH=$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}
                export DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode
                export HIP_DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode
                
                # Add ROCm binaries to PATH if not already there
                if [[ ":$PATH:" != *":$ROCM_PATH/bin:"* ]]; then
                    export PATH=$ROCM_PATH/bin:$PATH
                fi
            fi
            
            # Print configuration summary
            print_info "ROCm Configuration Summary:"
            print_info "  - PYTORCH_ROCM_ARCH: ${PYTORCH_ROCM_ARCH}"
            print_info "  - ROCBLAS_USE_HIPBLASLT: ${ROCBLAS_USE_HIPBLASLT}"
            print_info "  - HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION}"
            print_info "  - AMD_SERIALIZE_KERNEL: ${AMD_SERIALIZE_KERNEL}"
            print_info "  - TORCH_USE_HIP_DSA: ${TORCH_USE_HIP_DSA}"
            print_info "  - HIP_LAUNCH_BLOCKING: ${HIP_LAUNCH_BLOCKING}"
            print_info "  - HSA_ENABLE_SDMA: ${HSA_ENABLE_SDMA}"
            print_info "  - PYTORCH_ROCM_ALLOC_CONF: ${PYTORCH_ROCM_ALLOC_CONF}"
        fi
        
        # Verify GPU is available
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            DEVICE_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            print_success "GPU detected: $DEVICE_NAME"
        else
            print_warning "No GPU detected. Training will run on CPU (very slow)."
        fi
    else
        print_warning "PyTorch not available. Make sure it's installed."
        BACKEND="UNKNOWN"
    fi
}

# Function to check and activate virtual environment
setup_venv() {
    # Check if we're already in a virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        print_info "Virtual environment already active: $VIRTUAL_ENV"
        return 0
    fi
    
    # Check for uv venv
    if [ -d ".venv" ]; then
        print_info "Activating uv virtual environment..."
        source .venv/bin/activate
        print_success "Virtual environment activated"
        return 0
    fi
    
    # Check for standard venv
    if [ -d "venv" ]; then
        print_info "Activating virtual environment..."
        source venv/bin/activate
        print_success "Virtual environment activated"
        return 0
    fi
    
    print_warning "No virtual environment found. Using system Python."
    print_info "Consider creating one with: uv venv or python -m venv venv"
}

# Function to verify dependencies
verify_dependencies() {
    print_info "Verifying dependencies..."
    
    if ! python3 -c "import torch" 2>/dev/null; then
        print_error "PyTorch is not installed!"
        print_info "For ROCm: Run setup_rocm.sh or install manually"
        print_info "For CUDA: pip install torch torchvision torchaudio"
        exit 1
    fi
    
    # Check for other critical dependencies
    MISSING_DEPS=()
    for dep in "accelerate" "diffusers" "transformers"; do
        if ! python3 -c "import $dep" 2>/dev/null; then
            MISSING_DEPS+=("$dep")
        fi
    done
    
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        print_warning "Missing dependencies: ${MISSING_DEPS[*]}"
        print_info "Install with: pip install -r requirements.txt"
    else
        print_success "Core dependencies verified"
    fi
}

# Parse arguments
parse_args() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    MODE=$1
    shift
    
    case "$MODE" in
        train)
            if [ $# -eq 0 ]; then
                print_error "No config file specified for training mode"
                show_usage
                exit 1
            fi
            # Don't set CONFIG_FILE here - collect it after parsing options
            ;;
        gradio|ui|help)
            # No additional args needed for these modes
            ;;
        *)
            print_error "Unknown mode: $MODE"
            show_usage
            exit 1
            ;;
    esac
    
    # Initialize UI_DEV_MODE flag
    UI_DEV_MODE=false
    
    # Parse remaining options and collect config files
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--recover)
                RECOVER=true
                shift
                ;;
            -n|--name)
                JOB_NAME="$2"
                shift 2
                ;;
            -l|--log)
                LOG_FILE="$2"
                shift 2
                ;;
            -p|--port)
                UI_PORT="$2"
                shift 2
                ;;
            --dev)
                if [ "$MODE" = "ui" ]; then
                    UI_DEV_MODE=true
                else
                    print_warning "--dev flag is only valid for 'ui' mode"
                fi
                shift
                ;;
            *)
                # For train mode, remaining args are config files
                if [ "$MODE" = "train" ]; then
                    if [ -z "$CONFIG_FILE" ]; then
                        CONFIG_FILE="$1"
                    else
                        CONFIG_FILE="$CONFIG_FILE $1"
                    fi
                fi
                shift
                ;;
        esac
    done
    
    # Validate config file for train mode
    if [ "$MODE" = "train" ] && [ -z "$CONFIG_FILE" ]; then
        print_error "No config file specified for training mode"
        show_usage
        exit 1
    fi
}

# Main execution
main() {
    print_info "AI Toolkit Startup Script"
    print_info "=========================="
    echo ""
    
    # Parse arguments
    parse_args "$@"
    
    # Setup environment
    setup_venv
    detect_backend
    verify_dependencies
    
    echo ""
    print_info "Starting in mode: $MODE"
    echo ""
    
    # Execute based on mode
    case "$MODE" in
        train)
            print_info "Running training job(s)..."
            print_warning "⚠️  System safeguards will check for desktop environment and memory."
            print_info "If running on a desktop system, you'll be asked to confirm before training starts."
            print_info "See DESKTOP_SAFETY.md for details on protecting your system."
            CMD="python run.py"
            
            # Add config files
            for config in $CONFIG_FILE; do
                CMD="$CMD \"$config\""
            done
            
            # Add options
            if [ "$RECOVER" = true ]; then
                CMD="$CMD --recover"
            fi
            
            if [ -n "$JOB_NAME" ]; then
                CMD="$CMD --name \"$JOB_NAME\""
            fi
            
            if [ -n "$LOG_FILE" ]; then
                CMD="$CMD --log \"$LOG_FILE\""
            fi
            
            print_info "Command: $CMD"
            eval $CMD
            ;;
            
        gradio)
            print_info "Launching Gradio UI..."
            if ! python3 -c "import gradio" 2>/dev/null; then
                print_error "Gradio is not installed!"
                print_info "Install with: pip install gradio"
                exit 1
            fi
            python flux_train_ui.py
            ;;
            
        ui)
            print_info "Launching web UI on port $UI_PORT..."
            if [ ! -d "ui" ]; then
                print_error "UI directory not found!"
                exit 1
            fi
            
            cd ui
            if [ ! -d "node_modules" ]; then
                print_info "Installing UI dependencies..."
                npm install
            fi
            
            # Check if --dev flag is set for development mode with hot reload
            if [ "$UI_DEV_MODE" = "true" ]; then
                print_info "Starting UI in DEVELOPMENT mode (hot reload enabled)..."
                print_info "UI will be available at http://localhost:3000 (or next available port)"
                PORT=$UI_PORT npm run dev
            else
                print_info "Starting UI in PRODUCTION mode..."
                print_info "To use dev mode with hot reload, run: ./start_toolkit.sh ui --dev"
                PORT=$UI_PORT npm run build_and_start
            fi
            ;;
            
        help)
            show_usage
            ;;
    esac
}

# Run main function
main "$@"

