#!/bin/bash
# Run Vision-Language Model on RK3588 NPU
# Usage: ~/run-vlm.sh <image_path> [prompt]

MODEL_DIR=~/models
DEMO=~/rknn-llm/examples/multimodal_model_demo/deploy/build/demo

LLM_MODEL="$MODEL_DIR/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm"
VISION_MODEL="$MODEL_DIR/qwen3-vl-2b_vision_448_rk3588.rknn"
MAX_TOKENS="${3:-512}"
MAX_CTX="${4:-2048}"

IMAGE_PATH="${1}"
PROMPT="${2:-What is in the image?}"

if [ -z "$IMAGE_PATH" ]; then
    echo "Usage: ~/run-vlm.sh <image_path> [prompt]"
    echo ""
    echo "Examples:"
    echo "  ~/run-vlm.sh photo.jpg"
    echo "  ~/run-vlm.sh photo.jpg \"What objects are in this image?\""
    exit 1
fi

if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image not found: $IMAGE_PATH"
    exit 1
fi

echo "============================================"
echo "VLM: Qwen3-VL-2B | Image: $IMAGE_PATH"
echo "============================================"

# The <image> tag tells the model to use the loaded image
printf "<image>${PROMPT}\nexit\n" | $DEMO "$IMAGE_PATH" "$VISION_MODEL" "$LLM_MODEL" "$MAX_TOKENS" "$MAX_CTX" 3
