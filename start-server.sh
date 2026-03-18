#!/bin/bash
# Start RKLLM API server (OpenAI-compatible endpoint)
# Usage: ~/start-server.sh [model_name] [port]
#   model_name: qwen3-1.7b (default), qwen3-4b
#   port: default 8080

MODEL_NAME="${1:-qwen3-1.7b}"
PORT="${2:-8080}"
MODEL_DIR=~/models

case "$MODEL_NAME" in
    qwen3-1.7b|1.7b)
        MODEL_FILE="$MODEL_DIR/Qwen3-1.7B-w8a8-rk3588.rkllm"
        ;;
    qwen3-4b|4b)
        MODEL_FILE="$MODEL_DIR/Qwen3-4B-w8a8-rk3588.rkllm"
        ;;
    *)
        MODEL_FILE="$MODEL_NAME"
        ;;
esac

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model not found: $MODEL_FILE"
    ls -lh "$MODEL_DIR"/*.rkllm 2>/dev/null
    exit 1
fi

IP=$(hostname -I | awk '{print $1}')
echo "============================================"
echo "RKLLM API Server"
echo "Model: $(basename $MODEL_FILE)"
echo "Endpoint: http://$IP:$PORT/rkllm_chat"
echo "============================================"
echo ""
echo "Example request:"
echo "  curl -X POST http://$IP:$PORT/rkllm_chat \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"stream\":false,\"enable_thinking\":false,\"tools\":null}'"
echo ""
echo "Press Ctrl+C to stop."
echo ""

cd ~/rkllm-server
export PATH="$HOME/.local/bin:$PATH"
python3 flask_server.py \
    --rkllm_model_path "$MODEL_FILE" \
    --target_platform rk3588
