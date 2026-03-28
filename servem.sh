#!/bin/bash

# vLLM Model Serving Script
# Usage: ./servem.sh {MODEL_NAME}

# Save original state
ORIGINAL_WORK_DIR=$(pwd)
ORIGINAL_VENV_PATH="$VIRTUAL_ENV"  # Will be empty if no venv active

# Trap to ensure cleanup on exit
cleanup() {
    # Return to original directory
    cd "$ORIGINAL_WORK_DIR" 2>/dev/null || true

    # Restore original virtual environment
    if [ -n "$ORIGINAL_VENV_PATH" ]; then
        # Reactivate the original venv
        source "$ORIGINAL_VENV_PATH/bin/activate"
    elif [ -n "$VIRTUAL_ENV" ]; then
        # Deactivate if we activated vllm venv
        deactivate
    fi
}

# Set trap for cleanup on exit or interrupt
trap cleanup EXIT INT TERM

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to find next available port
find_available_port() {
    local start_port=$1
    local port=$start_port
    while check_port $port; do
        port=$((port + 1))
    done
    echo $port
}

# Model configurations
declare -A MODEL_PATHS
declare -A MODEL_PORTS
declare -A MODEL_GPUS
declare -A MODEL_MAX_LEN
declare -A MODEL_TP_SIZE
declare -A MODEL_GPU_UTIL

# Text models
MODEL_PATHS[llama3-8b]="/ssd_data/models/llama3-8b-instruct"
MODEL_PORTS[llama3-8b]=8000
MODEL_GPUS[llama3-8b]=0
MODEL_MAX_LEN[llama3-8b]=8192
MODEL_TP_SIZE[llama3-8b]=1
MODEL_GPU_UTIL[llama3-8b]=0.9

MODEL_PATHS[qwen3-4b]="/ssd_data/models/Qwen3-4B-Instruct-2507"
MODEL_PORTS[qwen3-4b]=8001
MODEL_GPUS[qwen3-4b]=1
MODEL_MAX_LEN[qwen3-4b]=32768
MODEL_TP_SIZE[qwen3-4b]=1
MODEL_GPU_UTIL[qwen3-4b]=0.9

MODEL_PATHS[qwen3-30b-fp8]="/ssd_data/models/Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_PORTS[qwen3-30b-fp8]=8004
MODEL_GPUS[qwen3-30b-fp8]="0,1"
MODEL_MAX_LEN[qwen3-30b-fp8]=32768
MODEL_TP_SIZE[qwen3-30b-fp8]=2
MODEL_GPU_UTIL[qwen3-30b-fp8]=0.8

# Vision model
MODEL_PATHS[qwen3-vl-8b]="/ssd_data/models/Qwen3-VL-8B-Instruct"
MODEL_PORTS[qwen3-vl-8b]=8002
MODEL_GPUS[qwen3-vl-8b]="2,3"
MODEL_MAX_LEN[qwen3-vl-8b]=32768
MODEL_TP_SIZE[qwen3-vl-8b]=2
MODEL_GPU_UTIL[qwen3-vl-8b]=0.9

MODEL_PATHS[llava-v1.6-7b]="/ssd_data/models/llava-v1___6-mistral-7b-hf"
MODEL_PORTS[llava-v1.6-7b]=8003
MODEL_GPUS[llava-v1.6-7b]="0,1"
MODEL_MAX_LEN[llava-v1.6-7b]=32768
MODEL_TP_SIZE[llava-v1.6-7b]=2
MODEL_GPU_UTIL[llava-v1.6-7b]=0.9

MODEL_PATHS[qwen3-vl-30b]="/ssd_data/models/Qwen3-VL-30B-A3B-Instruct-FP8"
MODEL_PORTS[qwen3-vl-30b]=8005
MODEL_GPUS[qwen3-vl-30b]="2,3"
MODEL_MAX_LEN[qwen3-vl-30b]=32768
MODEL_TP_SIZE[qwen3-vl-30b]=2
MODEL_GPU_UTIL[qwen3-vl-30b]=0.9

# Function to list available models
list_models() {
    echo ""
    echo "Available models:"
    echo "=================="
    for model in "${!MODEL_PATHS[@]}"; do
        if [ -d "${MODEL_PATHS[$model]}" ]; then
            echo -e "  ${GREEN}✓${NC} $model"
            echo "     Path: ${MODEL_PATHS[$model]}"
            echo "     Port: ${MODEL_PORTS[$model]} | GPUs: ${MODEL_GPUS[$model]} | Max Len: ${MODEL_MAX_LEN[$model]}"
            echo "     TP Size: ${MODEL_TP_SIZE[$model]} | GPU Util: ${MODEL_GPU_UTIL[$model]}"
            echo ""
        else
            echo -e "  ${RED}✗${NC} $model (not found at ${MODEL_PATHS[$model]})"
        fi
    done
}

# Function to serve a model
serve_model() {
    local model_key=$1
    local model_path=${MODEL_PATHS[$model_key]}
    local port=${MODEL_PORTS[$model_key]}
    local gpus=${MODEL_GPUS[$model_key]}
    local max_len=${MODEL_MAX_LEN[$model_key]}
    local tp_size=${MODEL_TP_SIZE[$model_key]}
    local gpu_util=${MODEL_GPU_UTIL[$model_key]}

    # Check if model exists
    if [ ! -d "$model_path" ]; then
        print_error "Model not found at: $model_path"
        echo "Available models:"
        list_models
        exit 1
    fi

    # Check if port is available
    if check_port $port; then
        print_warning "Port $port is already in use. Finding available port..."
        port=$(find_available_port $port)
        print_info "Using port $port instead"
    fi

    # Display configuration
    echo ""
    print_info "Starting vLLM server with configuration:"
    echo "  Model:       $model_key"
    echo "  Path:        $model_path"
    echo "  Port:        $port"
    echo "  GPUs:        $gpus"
    echo "  Max Length:  $max_len"
    echo "  TP Size:     $tp_size"
    echo "  GPU Util:    $gpu_util"
    echo ""

    # Activate vllm virtual environment
    print_info "Activating vllm virtual environment..."
    cd ~ && source venv/vllm/bin/activate

    # Set CUDA devices
    export CUDA_VISIBLE_DEVICES=$gpus

    # Construct command
    local cmd="vllm serve $model_path"
    cmd="$cmd --port $port"
    cmd="$cmd --max-model-len $max_len"
    cmd="$cmd --gpu-memory-utilization $gpu_util"

    # Add tensor parallelism if > 1
    if [ "$tp_size" -gt 1 ]; then
        cmd="$cmd --tensor-parallel-size $tp_size"
    fi

    # Print command
    print_info "Executing:"
    echo "  CUDA_VISIBLE_DEVICES=$gpus $cmd"
    echo ""

    # Execute command
    eval "CUDA_VISIBLE_DEVICES=$gpus $cmd"
}

# Main script logic
if [ $# -eq 0 ]; then
    echo "vLLM Model Serving Script"
    echo "=========================="
    echo ""
    echo "Usage: $0 <MODEL_NAME>"
    echo ""
    list_models
    exit 0
fi

MODEL_KEY=$(echo "$1" | tr '[:upper:]' '[:lower:]')

# Check if model exists in configuration
if [[ -v "MODEL_PATHS[$MODEL_KEY]" ]]; then
    serve_model "$MODEL_KEY"
else
    print_error "Unknown model: $1"
    echo ""
    echo "Available models:"
    list_models
    exit 1
fi
