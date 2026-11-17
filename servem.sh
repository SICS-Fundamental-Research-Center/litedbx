#!/usr/bin/env bash

set -euo pipefail

# vLLM model hosting scaffold.
#
# Supported commands:
#   ./servem.sh                 # list models
#   ./servem.sh list            # list models
#   ./servem.sh start MODEL     # stop prior session, probe GPUs, start in tmux
#   ./servem.sh swap MODEL      # stop ALL our sessions, then start MODEL (timed)
#   ./servem.sh stop MODEL      # stop tmux session and release resources
#   ./servem.sh status [MODEL]  # show session status
#   ./servem.sh state MODEL KEY  # print a saved state value
#
# Backward compatibility:
#   ./servem.sh <MODEL>         # same as start MODEL

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        print_error "Required command not found: $1"
        exit 1
    }
}

check_port() {
    local port=$1
    lsof -Pi ":$port" -sTCP:LISTEN -t >/dev/null 2>&1
}

wait_for_port_free() {
    local port=$1
    local timeout=${2:-30}
    local elapsed=0
    while check_port "$port"; do
        sleep 1
        elapsed=$((elapsed + 1))
        if (( elapsed >= timeout )); then
            return 1
        fi
    done
}

# The port is part of the CLIENT contract (llm/config.yaml hardcodes
# localhost:<port>): silently renumbering would report success while every
# client still points at the old port. Fail hard and show the squatter.
require_port_free() {
    local port=$1
    if ! check_port "$port"; then
        return 0
    fi
    print_error "Port $port is still in use -- refusing to renumber (client contract port):"
    lsof -Pi ":$port" -sTCP:LISTEN 2>/dev/null || true
    return 1
}

wait_for_http_ready() {
    local port=$1
    local timeout=${2:-600}
    local elapsed=0
    while true; do
        if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        if (( elapsed >= timeout )); then
            return 1
        fi
    done
}

model_api_matches() {
    local model_key=$1
    local port=$2
    local response

    command -v curl >/dev/null 2>&1 || return 1
    command -v python3 >/dev/null 2>&1 || return 1
    response=$(curl -sf --max-time 2 \
        "http://127.0.0.1:${port}/v1/models" 2>/dev/null) || return 1

    EXPECTED_MODEL_PATH="${MODEL_PATHS[$model_key]}" \
        RESPONSE="$response" python3 -c "import json, os; expected = os.environ[\"EXPECTED_MODEL_PATH\"]; models = json.loads(os.environ[\"RESPONSE\"]).get(\"data\", []); raise SystemExit(not any(model.get(\"id\") == expected or model.get(\"root\") == expected for model in models))" \
        >/dev/null 2>&1
}


trim() {
    local s=$1
    s=${s#"${s%%[![:space:]]*}"}
    s=${s%"${s##*[![:space:]]}"}
    echo "$s"
}

HOST_STATE_DIR="${HOST_STATE_DIR:-/tmp/litedbx-model-hosting}"
TMUX_PREFIX="${TMUX_PREFIX:-vllm}"
VLLM_VENV_PATH="${VLLM_VENV_PATH:-$HOME/venv/vllm}"
GPU_POOL_OVERRIDE="${SERVEM_GPU_POOL:-auto}"

mkdir -p "$HOST_STATE_DIR"

# Model configurations.
declare -A MODEL_PATHS
declare -A MODEL_PORTS
declare -A MODEL_MAX_LEN
declare -A MODEL_TP_SIZE
declare -A MODEL_GPU_UTIL
declare -A MODEL_MIN_FREE_MB
declare -A MODEL_EXTRA_ARGS

MODEL_PATHS[llama3-8b]="/ssd_data/models/llama3-8b-instruct"
MODEL_PORTS[llama3-8b]=8000
MODEL_MAX_LEN[llama3-8b]=8192
MODEL_TP_SIZE[llama3-8b]=1
MODEL_GPU_UTIL[llama3-8b]=0.9
MODEL_MIN_FREE_MB[llama3-8b]=12000

MODEL_PATHS[qwen3-4b]="/ssd_data/models/Qwen3-4B-Instruct-2507"
MODEL_PORTS[qwen3-4b]=8001
MODEL_MAX_LEN[qwen3-4b]=32768
MODEL_TP_SIZE[qwen3-4b]=1
MODEL_GPU_UTIL[qwen3-4b]=0.9
MODEL_MIN_FREE_MB[qwen3-4b]=8000

MODEL_PATHS[qwen3-30b-fp8]="/ssd_data/models/Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_PORTS[qwen3-30b-fp8]=8004
MODEL_MAX_LEN[qwen3-30b-fp8]=32768
MODEL_TP_SIZE[qwen3-30b-fp8]=2
# 0.8 (18.85 GiB) failed init_device on a GPU shared with another user's ~4.5 GiB
# job (only ~18.77 GiB free at measurement) -- "Free memory ... is less than
# desired GPU memory utilization". 0.75 (17.67 GiB) leaves ~1.1 GiB startup
# margin. gpu_memory_utilization only reserves KV-cache headroom -- it does NOT
# change model weights or outputs (temp=0 identical tokens), so this is hosting
# tuning, not result-tuning.
MODEL_GPU_UTIL[qwen3-30b-fp8]=0.75
MODEL_MIN_FREE_MB[qwen3-30b-fp8]=18000

MODEL_PATHS[qwen3-vl-8b]="/ssd_data/models/Qwen3-VL-8B-Instruct"
MODEL_PORTS[qwen3-vl-8b]=8002
MODEL_MAX_LEN[qwen3-vl-8b]=32768
MODEL_TP_SIZE[qwen3-vl-8b]=2
MODEL_GPU_UTIL[qwen3-vl-8b]=0.9
MODEL_MIN_FREE_MB[qwen3-vl-8b]=14000

MODEL_PATHS[llava-v1.6-7b]="/ssd_data/models/llava-v1___6-mistral-7b-hf"
MODEL_PORTS[llava-v1.6-7b]=8003
MODEL_MAX_LEN[llava-v1.6-7b]=32768
MODEL_TP_SIZE[llava-v1.6-7b]=2
MODEL_GPU_UTIL[llava-v1.6-7b]=0.9
MODEL_MIN_FREE_MB[llava-v1.6-7b]=12000

MODEL_PATHS[qwen3-vl-30b]="/ssd_data/models/Qwen3-VL-30B-A3B-Instruct-FP8"
MODEL_PORTS[qwen3-vl-30b]=8005
MODEL_MAX_LEN[qwen3-vl-30b]=32768
MODEL_TP_SIZE[qwen3-vl-30b]=2
# 0.8 leaves only ~1.48 GiB for KV cache, but max_model_len 32768 needs ~1.5 GiB
# per seq -- engine dies at init with "1.5 GiB KV cache is needed, which is
# larger than the available KV cache memory (1.48 GiB)". 0.9 gives ~3.9 GiB KV
# headroom (proven Jun-Aug 2026). If shared-box pressure blocks launch at 0.9,
# prefer dropping max_model_len to 12288 (validated VL recipe) over lowering
# util below ~0.85, which cannot serve 32k context on 24 GiB GPUs.
MODEL_GPU_UTIL[qwen3-vl-30b]=0.9
MODEL_MIN_FREE_MB[qwen3-vl-30b]=18000

MODEL_PATHS[qwen3-vl-2b]="/ssd_data/models/Qwen3-VL-2B-Instruct"
MODEL_PORTS[qwen3-vl-2b]=8006
MODEL_MAX_LEN[qwen3-vl-2b]=32768
MODEL_TP_SIZE[qwen3-vl-2b]=2
MODEL_GPU_UTIL[qwen3-vl-2b]=0.9
MODEL_MIN_FREE_MB[qwen3-vl-2b]=6000

MODEL_PATHS[qwen3-vl-4b]="/ssd_data/models/Qwen3-VL-4B-Instruct"
MODEL_PORTS[qwen3-vl-4b]=8007
MODEL_MAX_LEN[qwen3-vl-4b]=32768
MODEL_TP_SIZE[qwen3-vl-4b]=2
MODEL_GPU_UTIL[qwen3-vl-4b]=0.9
MODEL_MIN_FREE_MB[qwen3-vl-4b]=8000

# Audio(+vision) model, bf16 Instruct. The -FP8 checkpoint is NOT servable on
# this box (4x3090/Ampere + vllm 0.19.0): TP=2 -> Marlin shard 576 % block_k
# 128; PP=2 auto -> Marlin tile size_n 4304 % 64; PP=2 triton -> quant scheme
# unsupported; PP=2 cutlass -> disabled on SM86. bf16 needs TP=4 (~15GB
# weights/card). OOM note: 16-way concurrent AUDIO prefill spikes activations;
# keep util <= 0.75-0.80 or raise it only with audio-batch caps.
# 11-16 (R1.D1.1 audio arm): engine sem_map / timing replay fire one-shot
# gathers (client semaphore 20) -- the audio chunk-8 recipe is enforced
# SERVER-SIDE via max-num-seqs 8 so any client concurrency is safe.
MODEL_PATHS[qwen3-omni-30b]="/ssd_data/models/Qwen3-Omni-30B-A3B-Instruct"
MODEL_PORTS[qwen3-omni-30b]=8008
MODEL_MAX_LEN[qwen3-omni-30b]=32768
MODEL_TP_SIZE[qwen3-omni-30b]=4
MODEL_GPU_UTIL[qwen3-omni-30b]=0.75
MODEL_MIN_FREE_MB[qwen3-omni-30b]=18000
MODEL_EXTRA_ARGS[qwen3-omni-30b]="--max-num-seqs 8"

# R1.D3.1 Phase C (11-16): community AWQ W4A16 quant of the omni model
# (cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit, tower/talker left bf16,
# thinker-only AWQ -> TP2 fits ~13.8 GiB/card). Same audio OOM recipe as
# bf16: util 0.75 + server-side max-num-seqs 8.
# PATH IS TEMPORARY: served straight from the staging download dir because
# /ssd_data had only 21 GiB free (ckpt 27.57 GiB). Pending user ruling:
# promote to /ssd_data/models/ when space allows (update this path), or
# delete the ckpt (drop this entry) if AWQ is retired.
MODEL_PATHS[qwen3-omni-30b-awq]="/home/baiwenchao/models/staging/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"
MODEL_PORTS[qwen3-omni-30b-awq]=8009
MODEL_MAX_LEN[qwen3-omni-30b-awq]=32768
MODEL_TP_SIZE[qwen3-omni-30b-awq]=2
MODEL_GPU_UTIL[qwen3-omni-30b-awq]=0.75
# min-free 18000 = util 0.75 x 24576 MiB (18.4G) needs a card that free or
# better, else selection passes and vllm dies at init_device (the fp8 death
# mode documented above); matches the bf16 sibling entry.
MODEL_MIN_FREE_MB[qwen3-omni-30b-awq]=18000
MODEL_EXTRA_ARGS[qwen3-omni-30b-awq]="--max-num-seqs 8"

session_name() {
    local model_key=$1
    echo "$TMUX_PREFIX-$model_key"
}

state_file() {
    echo "$HOST_STATE_DIR/$1.env"
}

save_state() {
    local model_key=$1
    local port=$2
    local gpus=$3
    local session=$4
    local path=$(state_file "$model_key")

    cat > "$path" <<EOF
MODEL_KEY=$model_key
MODEL_PATH=${MODEL_PATHS[$model_key]}
MODEL_PORT=$port
CUDA_VISIBLE_DEVICES=$gpus
TMUX_SESSION=$session
EOF
}

load_state_value() {
    local model_key=$1
    local key=$2
    local path=$(state_file "$model_key")
    [[ -f "$path" ]] || return 1
    # shellcheck disable=SC1090
    source "$path"
    case "$key" in
        MODEL_KEY) echo "${MODEL_KEY:-}" ;;
        MODEL_PATH) echo "${MODEL_PATH:-}" ;;
        MODEL_PORT) echo "${MODEL_PORT:-}" ;;
        CUDA_VISIBLE_DEVICES) echo "${CUDA_VISIBLE_DEVICES:-}" ;;
        TMUX_SESSION) echo "${TMUX_SESSION:-}" ;;
        *) return 1 ;;
    esac
}

clear_state() {
    rm -f "$(state_file "$1")"
}

kill_pid_tree() {
    local root_pid=$1
    local signal=${2:-TERM}

    local child_pids
    child_pids=$(ps -eo pid=,ppid= | awk -v p="$root_pid" '$2 == p {print $1}') || true
    for child_pid in $child_pids; do
        kill_pid_tree "$child_pid" "$signal"
    done

    kill -s "$signal" "$root_pid" 2>/dev/null || true
}

probe_gpu_inventory() {
    require_cmd nvidia-smi
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
}

filter_gpu_pool() {
    local gpu_pool=$1
    local idx=$2
    [[ "$gpu_pool" == "auto" || -z "$gpu_pool" ]] && return 0
    [[ ",$gpu_pool," == *",$idx,"* ]]
}

print_gpu_probe() {
    local model_key=${1:-}
    local min_free_mb=0
    local tp_size=1
    if [[ -n "$model_key" ]]; then
        min_free_mb=${MODEL_MIN_FREE_MB[$model_key]}
        tp_size=${MODEL_TP_SIZE[$model_key]}
    fi

    echo "GPU inventory:"
    echo "=============================================================="
    printf "%-5s %-8s %-8s %-8s %-8s %s\n" "GPU" "FREE" "TOTAL" "USED" "UTIL" "NAME"
    while IFS=, read -r idx name total used free util; do
        idx=$(trim "$idx")
        name=$(trim "$name")
        total=$(trim "$total")
        used=$(trim "$used")
        free=$(trim "$free")
        util=$(trim "$util")
        local mark=""
        if [[ -n "$model_key" && "$free" -ge "$min_free_mb" ]]; then
            mark="eligible"
        elif [[ -n "$model_key" ]]; then
            mark="busy"
        fi
        printf "%-5s %-8s %-8s %-8s %-8s %s %s\n" "$idx" "$free" "$total" "$used" "$util" "$name" "$mark"
    done < <(probe_gpu_inventory)

    if [[ -n "$model_key" ]]; then
        echo "--------------------------------------------------------------"
        echo "Model: $model_key | TP size: $tp_size | Min free per GPU: ${min_free_mb}MB"
        if select_gpu_slots "$model_key" >/dev/null 2>&1; then
            echo "Selected GPUs: $(select_gpu_slots "$model_key")"
        else
            echo "Selected GPUs: none"
        fi
    fi
    echo "=============================================================="
}

select_gpu_slots() {
    local model_key=$1
    local tp_size=${MODEL_TP_SIZE[$model_key]}
    local min_free_mb=${MODEL_MIN_FREE_MB[$model_key]}
    local gpu_pool=$GPU_POOL_OVERRIDE
    local candidates

    candidates=$(probe_gpu_inventory | awk -F', ' -v pool="$gpu_pool" -v min_free="$min_free_mb" '
        function in_pool(id) {
            if (pool == "" || pool == "auto") return 1
            return index("," pool ",", "," id ",") > 0
        }
        {
            idx=$1; name=$2; total=$3+0; used=$4+0; free=$5+0; util=$6+0;
            if (in_pool(idx) && free >= min_free) {
                printf "%s\t%s\t%s\t%s\t%s\t%s\n", idx, free, total, used, util, name
            }
        }
    ' | sort -t$'\t' -k2,2nr)

    if [[ -z "$candidates" ]]; then
        return 1
    fi

    mapfile -t chosen < <(printf '%s\n' "$candidates" | head -n "$tp_size" | awk -F'\t' '{print $1}')
    if (( ${#chosen[@]} < tp_size )); then
        return 1
    fi

    local IFS=,
    echo "${chosen[*]}"
}

# Returns 1 (does not exit) so callers -- notably swap_model -- can run their
# own failure accounting before exiting.
ensure_model_exists() {
    local model_key=$1
    local model_path=${MODEL_PATHS[$model_key]}
    if [[ ! -d "$model_path" ]]; then
        print_error "Model not found at: $model_path"
        return 1
    fi
}

stop_model() {
    local model_key=$1
    local session
    session=$(session_name "$model_key")
    local session_gpus=""

    if tmux has-session -t "$session" 2>/dev/null; then
        local pane_pid
        pane_pid=$(tmux display-message -p -t "$session" '#{pane_pid}' 2>/dev/null || true)
        print_warning "Stopping tmux session: $session"
        if [[ -n "${pane_pid:-}" ]]; then
            print_warning "Killing process tree rooted at pane pid: $pane_pid"
            kill_pid_tree "$pane_pid" TERM
            sleep 2
            kill_pid_tree "$pane_pid" KILL
        fi
        tmux kill-session -t "$session" 2>/dev/null || true
    fi

    if [[ -f "$(state_file "$model_key")" ]]; then
        # shellcheck disable=SC1090
        source "$(state_file "$model_key")"
        session_gpus="${CUDA_VISIBLE_DEVICES:-}"
    fi

    # Final sweep for any orphaned vLLM worker processes for this model only.
    if [[ -n "$session_gpus" ]]; then
        pkill -TERM -f "VLLM::Worker_.*CUDA_VISIBLE_DEVICES=${session_gpus//,/.*}" 2>/dev/null || true
        sleep 1
        pkill -KILL -f "VLLM::Worker_.*CUDA_VISIBLE_DEVICES=${session_gpus//,/.*}" 2>/dev/null || true
    fi

    clear_state "$model_key"
}

start_model() {
    local model_key=$1
    local model_path=${MODEL_PATHS[$model_key]}
    local port=${MODEL_PORTS[$model_key]}
    local max_len=${MODEL_MAX_LEN[$model_key]}
    local tp_size=${MODEL_TP_SIZE[$model_key]}
    local gpu_util=${MODEL_GPU_UTIL[$model_key]}
    local session
    local gpus

    require_cmd tmux
    require_cmd lsof
    require_cmd nvidia-smi

    # Failure paths below RETURN 1 (not exit): swap_model must survive them
    # to write its timing row (status=load-failed) before exiting.
    ensure_model_exists "$model_key" || return 1

    # Kill the previous hosting session for the same model first, then verify
    # the contract port is actually free (hard fail -- see require_port_free).
    stop_model "$model_key" || true
    wait_for_port_free "$port" 30 || true
    require_port_free "$port" || return 1

    gpus=$(select_gpu_slots "$model_key") || {
        print_error "No GPU slot has enough free memory for $model_key"
        print_gpu_probe "$model_key"
        return 1
    }

    session=$(session_name "$model_key")
    local -a cmd=(vllm serve "$model_path" --port "$port" --max-model-len "$max_len" --gpu-memory-utilization "$gpu_util")
    if (( tp_size > 1 )); then
        cmd+=(--tensor-parallel-size "$tp_size")
    fi
    if [[ -n "${MODEL_EXTRA_ARGS[$model_key]:-}" ]]; then
        read -ra extra_args <<< "${MODEL_EXTRA_ARGS[$model_key]}"
        cmd+=("${extra_args[@]}")
    fi
    # Shared-box accommodation: skip CUDA-graph capture (enforce eager) so a TP
    # shard on a fragmented/contended GPU does not fault during graph capture.
    # Env-gated (SERVEM_EAGER=1); identical model outputs (only ~10-20% slower).
    if [[ "${SERVEM_EAGER:-0}" == "1" ]]; then
        cmd+=(--enforce-eager)
    fi

    local launch_cmd
    printf -v launch_cmd 'source %q && export CUDA_VISIBLE_DEVICES=%q && exec ' "$VLLM_VENV_PATH/bin/activate" "$gpus"
    printf -v launch_cmd '%s%q ' "$launch_cmd" "${cmd[0]}"
    for ((i=1; i<${#cmd[@]}; i++)); do
        printf -v launch_cmd '%s%q ' "$launch_cmd" "${cmd[$i]}"
    done
    # Persist server output: a silently-dead tmux session otherwise leaves no
    # trace of the crash.
    printf -v launch_cmd '%s >> %q 2>&1' "$launch_cmd" "/tmp/servem-${model_key}.log"

    print_info "Starting model in tmux session: $session"
    print_info "CUDA_VISIBLE_DEVICES=$gpus"
    print_info "Port: $port"
    tmux new-session -d -s "$session" bash -lc "$launch_cmd"
    save_state "$model_key" "$port" "$gpus" "$session"

    print_info "Waiting for model to become ready on port $port..."
    if ! wait_for_http_ready "$port" 600; then
        print_error "Model failed to become ready on port $port"
        stop_model "$model_key" || true
        return 1
    fi

    print_success "Model started. Attach with: tmux attach -t $session"
    print_info "API: http://localhost:$port/v1"
}

# Coarse-grained serving swap -- stop ALL of OUR vllm-* sessions, then start
# MODEL -- with two-segment wall-clock timing:
#   evict segment = swap issued -> all our tmux sessions gone
#   load  segment = start issued -> HTTP /v1/models ready
# Timing rows are appended (never rewritten) to $SERVEM_SWAP_LOG -- a
# NEUTRAL default (/tmp/servem-swap.log); experiments that need a ledger set
# SERVEM_SWAP_LOG themselves (R1.D3.1 used
# revision/exp/R1.D3.1/2-batch-run/swap_raw.log). The already-serving
# short-circuit records a zero-cost row without evicting.
our_vllm_sessions() {
    tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^vllm-' || true
}

swap_model() {
    local model_key=$1
    local raw_log="${SERVEM_SWAP_LOG:-/tmp/servem-swap.log}"
    local evict0 evict1 load0 load1 stopped="" status=ok

    # Short-circuit: target already served and healthy -> record and return.
    # (swap semantics = "ensure MODEL is served"; no pointless evict+reload)
    local port="${MODEL_PORTS[$model_key]:-}"
    if [[ -n "$port" && -f "$HOST_STATE_DIR/${model_key}.env" ]] \
        && tmux has-session -t "vllm-$model_key" 2>/dev/null \
        && curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
        mkdir -p "$(dirname "$raw_log")"
        [[ -f "$raw_log" ]] || printf 'ts\tmodel\tstopped\tevict_start\tevict_end\tload_start\tload_end\tevict_s\tload_s\tstatus\n' >> "$raw_log"
        local now=$(date +%s.%N)
        printf '%s\t%s\tnone\t%s\t%s\t%s\t%s\t0.000\t0.000\talready-serving\n' \
            "$(date -Is)" "$model_key" "$now" "$now" "$now" "$now" >> "$raw_log"
        print_success "swap: $model_key already serving on :$port (recorded, no-op)"
        return 0
    fi

    evict0=$(date +%s.%N)
    # Stop every model we host (state files are the ground truth for "ours"),
    # then sweep any orphan vllm-* sessions by prefix.
    local state_dir="$HOST_STATE_DIR"
    local f key
    for f in "$state_dir"/*.env; do
        [[ -f "$f" ]] || continue
        key=$(grep -m1 '^MODEL_KEY=' "$f" | cut -d= -f2-)
        [[ -n "$key" ]] || continue
        stopped+="${stopped:+,}$key"
        stop_model "$key" || true
    done
    local s
    for s in $(our_vllm_sessions); do
        stopped+="${stopped:+,}${s#vllm-}(session-only)"
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    # Eviction completes when no vllm-* session remains (bounded wait).
    local waited=0
    while [[ -n "$(our_vllm_sessions)" ]]; do
        sleep 1
        waited=$((waited + 1))
        if (( waited >= 120 )); then
            status="evict-timeout"
            break
        fi
    done
    evict1=$(date +%s.%N)
    load0=$evict1

    if ! start_model "$model_key"; then
        status="load-failed"
    fi
    load1=$(date +%s.%N)

    mkdir -p "$(dirname "$raw_log")"
    [[ -f "$raw_log" ]] || printf 'ts\tmodel\tstopped\tevict_start\tevict_end\tload_start\tload_end\tevict_s\tload_s\tstatus\n' >> "$raw_log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%s\n' \
        "$(date -Is)" "$model_key" "${stopped:-none}" \
        "$evict0" "$evict1" "$load0" "$load1" \
        "$(echo "$evict1 $evict0" | awk '{print $1-$2}')" \
        "$(echo "$load1 $load0" | awk '{print $1-$2}')" \
        "$status" >> "$raw_log"
    [[ "$status" == ok ]] || exit 1
}

status_model() {
    local model_key=${1:-}
    if [[ -z "$model_key" ]]; then
        echo "Hosted models:"
        echo "==============="
        for model_key in "${!MODEL_PATHS[@]}"; do
            status_model "$model_key"
        done | sed "/^$/d"
        return 0
    fi

    local state
    state=$(state_file "$model_key")
    local session
    session=$(session_name "$model_key")
    local port=${MODEL_PORTS[$model_key]}
    local gpus=""

    if [[ -f "$state" ]]; then
        # shellcheck disable=SC1090
        source "$state"
        session=${TMUX_SESSION:-$session}
        port=${MODEL_PORT:-$port}
        gpus=${CUDA_VISIBLE_DEVICES:-}
    fi

    local has_session=false
    if tmux has-session -t "$session" 2>/dev/null; then
        has_session=true
    fi

    if model_api_matches "$model_key" "$port"; then
        if [[ "$has_session" == true ]]; then
            print_success "$model_key is running in tmux session $session"
        else
            print_success "$model_key is running (tmux session not found)"
        fi
    elif [[ "$has_session" == true ]]; then
        print_warning "$model_key is starting in tmux session $session"
    else
        print_warning "$model_key is not running"
    fi

    echo "  Port: $port"
    if [[ -n "$gpus" ]]; then
        echo "  GPUs: $gpus"
    fi
}

list_models() {
    echo ""
    echo "Available models:"
    echo "=================="
    for model in "${!MODEL_PATHS[@]}"; do
        if [[ -d "${MODEL_PATHS[$model]}" ]]; then
            echo -e "  ${GREEN}✓${NC} $model"
            echo "     Path: ${MODEL_PATHS[$model]}"
            echo "     Port: ${MODEL_PORTS[$model]} | GPU policy: auto | Max Len: ${MODEL_MAX_LEN[$model]}"
            echo "     TP Size: ${MODEL_TP_SIZE[$model]} | GPU Util: ${MODEL_GPU_UTIL[$model]} | Min Free: ${MODEL_MIN_FREE_MB[$model]}MB"
            echo ""
        else
            echo -e "  ${RED}✗${NC} $model (not found at ${MODEL_PATHS[$model]})"
        fi
    done
}

usage() {
    cat <<EOF
vLLM Model Hosting Scaffold
===========================

Usage:
  $0                    # list models
  $0 list               # list models
  $0 start MODEL        # stop previous session, probe GPUs, start in tmux
  $0 swap MODEL         # stop ALL our sessions, then start MODEL (timed;
                        #   raw rows appended to ${SERVEM_SWAP_LOG:-/tmp/servem-swap.log})
  $0 stop MODEL        # stop tmux session and clear state
  $0 status [MODEL]    # show session status
  $0 state MODEL KEY   # print a saved state value

Backwards compatible:
  $0 <MODEL>
EOF
}

main() {
    local cmd=${1:-list}
    case "$cmd" in
        list)
            list_models
            ;;
        start)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            local model_key
            model_key=$(echo "$2" | tr '[:upper:]' '[:lower:]')
            [[ -v "MODEL_PATHS[$model_key]" ]] || { print_error "Unknown model: $2"; exit 1; }
            start_model "$model_key"
            ;;
        stop)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            local model_key
            model_key=$(echo "$2" | tr '[:upper:]' '[:lower:]')
            [[ -v "MODEL_PATHS[$model_key]" ]] || { print_error "Unknown model: $2"; exit 1; }
            stop_model "$model_key"
            ;;
        swap)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            local model_key
            model_key=$(echo "$2" | tr '[:upper:]' '[:lower:]')
            [[ -v "MODEL_PATHS[$model_key]" ]] || { print_error "Unknown model: $2"; exit 1; }
            swap_model "$model_key"
            ;;
        status)
            if [[ $# -ge 2 ]]; then
                local model_key
                model_key=$(echo "$2" | tr '[:upper:]' '[:lower:]')
                [[ -v "MODEL_PATHS[$model_key]" ]] || { print_error "Unknown model: $2"; exit 1; }
                status_model "$model_key"
            else
                status_model
            fi
            ;;
        state)
            [[ -n "$3" ]] || { usage; exit 1; }
            local model_key
            model_key=$(echo "$2" | tr '[:upper:]' '[:lower:]')
            [[ -v "MODEL_PATHS[$model_key]" ]] || { print_error "Unknown model: $2"; exit 1; }
            load_state_value "$model_key" "$3"
            ;;
        help|-h|--help)
            usage
            ;;
        *)
            local model_key
            model_key=$(echo "$cmd" | tr '[:upper:]' '[:lower:]')
            if [[ -v "MODEL_PATHS[$model_key]" ]]; then
                start_model "$model_key"
            else
                print_error "Unknown command or model: $cmd"
                usage
                exit 1
            fi
            ;;
    esac
}

main "$@"
