#!/bin/bash
# Run LLM interactively on RK3588 NPU
# Usage: ~/run-llm.sh [model_name] [max_new_tokens] [max_context_len]

MODEL_DIR=~/models
DEMO=~/rknn-llm/examples/rkllm_api_demo/deploy/build/llm_demo

MODEL_NAME="${1:-qwen3-1.7b}"
MAX_TOKENS="${2:-512}"
MAX_CTX="${3:-2048}"

case "$MODEL_NAME" in
    qwen3-1.7b|1.7b)
        MODEL_FILE="$MODEL_DIR/Qwen3-1.7B-w8a8-rk3588.rkllm"
        ;;
    qwen3-4b|4b)
        MODEL_FILE="$MODEL_DIR/Qwen3-4B-w8a8-rk3588.rkllm"
        MAX_CTX="${3:-4096}"
        ;;
    *)
        MODEL_FILE="$MODEL_NAME"
        ;;
esac

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model not found: $MODEL_FILE"
    echo ""
    echo "Available models:"
    ls -lh "$MODEL_DIR"/*.rkllm 2>/dev/null | awk '{printf "  %-50s %s\n", $NF, $5}'
    exit 1
fi

echo "============================================"
echo "Model: $(basename $MODEL_FILE)"
echo "Max tokens: $MAX_TOKENS | Max context: $MAX_CTX"
echo "Type your question and press Enter."
echo "Type 'exit' to quit."
echo "============================================"
$DEMO "$MODEL_FILE" "$MAX_TOKENS" "$MAX_CTX"
