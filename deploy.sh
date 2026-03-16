#!/bin/bash
# Deploy RKLLM Web App to ROCK 5B
# Usage: bash deploy.sh [rock-5b-hostname]
#
# This script:
# 1. Copies server files to the board
# 2. Builds the VLM stream binary
# 3. Installs Python dependencies
# 4. Copies the NPU frequency fix script
# 5. Prints instructions to start the server

set -e

HOST="${1:-rock@rock-5b}"
REMOTE_DIR="~/rkllm-web"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== RKLLM Web App Deployment ==="
echo "Target: $HOST:$REMOTE_DIR"
echo ""

# 1. Create remote directory and copy files
echo "[1/5] Copying files to board..."
ssh "$HOST" "mkdir -p $REMOTE_DIR"
scp "$SCRIPT_DIR/server.py" "$HOST:$REMOTE_DIR/"
scp "$SCRIPT_DIR/index.html" "$HOST:$REMOTE_DIR/"
scp "$SCRIPT_DIR/vlm_stream.cpp" "$HOST:$REMOTE_DIR/"
scp "$SCRIPT_DIR/CMakeLists.txt" "$HOST:$REMOTE_DIR/"
echo "  Files copied."

# 2. Copy supporting files
echo "[2/5] Setting up dependencies..."
ssh "$HOST" "
  # Copy RKLLM runtime library if not already in /usr/lib
  if [ ! -f /usr/lib/librkllmrt.so ]; then
    echo 'rock' | sudo -S cp ~/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so /usr/lib/
    echo 'rock' | sudo -S ldconfig
  fi

  # Copy freq fix script
  cp ~/rknn-llm/scripts/fix_freq_rk3588.sh $REMOTE_DIR/ 2>/dev/null || true

  # Install Python dependencies
  python3 -c 'import flask' 2>/dev/null || pip3 install --break-system-packages flask
  python3 -c 'import onnxruntime' 2>/dev/null || pip3 install --break-system-packages onnxruntime opencv-python-headless pillow
"
echo "  Dependencies ready."

# 3. Build VLM stream binary
echo "[3/5] Building VLM stream binary..."
ssh "$HOST" "
  cd $REMOTE_DIR
  mkdir -p build && cd build
  cmake .. 2>&1 | tail -3
  make -j4 2>&1 | tail -5
  cp vlm_stream $REMOTE_DIR/ 2>/dev/null || true
"
echo "  VLM binary built."

# 4. Verify everything
echo "[4/5] Verifying deployment..."
ssh "$HOST" "
  echo 'Checking files...'
  ls -la $REMOTE_DIR/server.py $REMOTE_DIR/index.html $REMOTE_DIR/vlm_stream
  echo ''
  echo 'Checking models...'
  ls -lh ~/models/*.rkllm ~/models/*.rknn ~/models/sam2.1/* 2>/dev/null
  echo ''
  echo 'Checking memory...'
  free -h | head -2
"

# 5. Create systemd service (optional)
echo "[5/5] Creating start script..."
ssh "$HOST" "cat > $REMOTE_DIR/start.sh << 'SCRIPT'
#!/bin/bash
# Start RKLLM Web Server
# Usage: ~/rkllm-web/start.sh [--preload qwen3-1.7b]

cd ~/rkllm-web
export PATH=\"\$HOME/.local/bin:\$PATH\"

echo \"Stopping any existing server...\"
pkill -f 'python3 server.py' 2>/dev/null
sleep 1

echo \"Starting RKLLM Web Server...\"
python3 server.py \"\$@\"
SCRIPT
chmod +x $REMOTE_DIR/start.sh
"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "To start the server, SSH into the board and run:"
echo ""
echo "  ssh $HOST"
echo "  ~/rkllm-web/start.sh"
echo ""
echo "Or with a preloaded model:"
echo ""
echo "  ~/rkllm-web/start.sh --preload qwen3-1.7b"
echo ""
echo "Then open in your browser:"
echo ""
echo "  http://rock-5b:8080"
echo ""
