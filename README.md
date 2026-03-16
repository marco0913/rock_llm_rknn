# Running LLMs & VLMs on ROCK 5B (RK3588 NPU)

Complete guide to running local AI models on the Radxa ROCK 5B (16GB RAM) using the RK3588's built-in NPU with RKLLM runtime.

## Hardware

- **Board:** Radxa ROCK 5B
- **SoC:** RK3588 (4x A76 + 4x A55 cores, 6 TOPS NPU)
- **RAM:** 16GB LPDDR4x
- **Storage:** 32GB microSD card
- **NPU:** 3-core RKNPU2, used for all model inference

## What We're Running

| Model | Size | Type | Use Case |
|-------|------|------|----------|
| Qwen3 1.7B (W8A8) | 2.3 GB | Text LLM | Fast chat, simple tasks |
| Qwen3 4B (W8A8) | 5.0 GB | Text LLM | Smarter reasoning, thinking mode |
| Qwen3-VL 2B (W8A8) | 2.3 GB + 850 MB vision encoder | Vision-Language | Image analysis, live camera stream |
| SAM 2.1 Small | 358 MB encoder + 16 MB decoder | Segmentation | Click-to-segment objects, live stream segmentation |

LLM/VLM models run entirely on the NPU. SAM 2.1 encoder runs on NPU, decoder on CPU.

---

## Step 1: Flash the OS

The stock Debian 11 (Bullseye) image ships with RKNPU driver **0.8.2**, which is too old for RKLLM (needs >= 0.9.7). The Radxa apt repos for Bullseye are dead (404s on .deb files), so upgrading the kernel in-place is not possible.

**Solution:** Flash a newer image.

1. Download `rock-5b_bookworm_kde_b5.output.img.xz` (1.44 GB) from:
   https://github.com/radxa-build/rock-5b/releases/tag/rsdk-b5
2. Flash to a microSD card using balenaEtcher, Rufus, or `dd`
3. Boot the ROCK 5B from the new card

This gives you:
- **Debian 12 (Bookworm)**
- **Kernel 6.1.43** with RKNPU driver **0.9.6**

Default credentials: `rock` / `rock`

## Step 2: Upgrade Kernel & NPU Driver

The flashed image has driver 0.9.6, but RKLLM needs >= 0.9.7. Fortunately, the Bookworm repos work:

```bash
sudo apt update
sudo apt install -y linux-image-rock-5b rknpu2-rk3588 cmake g++
```

This upgrades to:
- **Kernel 6.1.84** with RKNPU driver **0.9.8**
- **RKNPU2 userspace 2.3.0**

> Note: The `radxa-overlays` DKMS module may fail to build — this is harmless. The kernel and RKNPU driver install correctly regardless.

Reboot:
```bash
sudo reboot
```

Verify after reboot:
```bash
uname -r
# Expected: 6.1.84-8-rk2410

dmesg | grep rknpu
# Look for: [drm] Initialized rknpu 0.9.8
```

## Step 3: Clone RKLLM and Install Runtime

```bash
cd ~
git clone https://github.com/airockchip/rknn-llm.git
cd rknn-llm
git checkout release-v1.2.3

# Install the RKLLM runtime library
sudo cp rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so /usr/lib/
sudo ldconfig
```

## Step 4: Build the Text LLM Demo

```bash
cd ~/rknn-llm/examples/rkllm_api_demo/deploy
mkdir -p build && cd build
cmake ..
make -j4
```

Binary is at: `~/rknn-llm/examples/rkllm_api_demo/deploy/build/llm_demo`

## Step 5: Build the Multimodal (VLM) Demo

```bash
cd ~/rknn-llm/examples/multimodal_model_demo/deploy
mkdir -p build && cd build
cmake ..
make -j4
```

Binary is at: `~/rknn-llm/examples/multimodal_model_demo/deploy/build/demo`

## Step 6: Download Models

```bash
mkdir -p ~/models

# Qwen3 1.7B — fast text model (2.3 GB)
wget -O ~/models/Qwen3-1.7B-w8a8-rk3588.rkllm \
  'https://huggingface.co/GatekeeperZA/Qwen3-1.7B-RKLLM-v1.2.3/resolve/main/Qwen3-1.7B-w8a8-rk3588.rkllm'

# Qwen3 4B — smarter text model (5.0 GB)
wget -O ~/models/Qwen3-4B-w8a8-rk3588.rkllm \
  'https://huggingface.co/GatekeeperZA/Qwen3-4B-RKLLM-v1.2.3/resolve/main/Qwen3-4B-w8a8-rk3588.rkllm'

# Qwen3-VL 2B — vision-language LLM part (2.3 GB)
wget -O ~/models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm \
  'https://huggingface.co/GatekeeperZA/Qwen3-VL-2B-Instruct-RKLLM-v1.2.3/resolve/main/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm'

# Qwen3-VL 2B — vision encoder (850 MB)
wget -O ~/models/qwen3-vl-2b_vision_448_rk3588.rknn \
  'https://huggingface.co/GatekeeperZA/Qwen3-VL-2B-Instruct-RKLLM-v1.2.3/resolve/main/qwen3-vl-2b_vision_448_rk3588.rknn'

# SAM 2.1 Small — segmentation encoder (358 MB, runs on NPU)
mkdir -p ~/models/sam2.1
wget -O ~/models/sam2.1/sam2.1_hiera_small_encoder.rknn \
  'https://huggingface.co/happyme531/Segment-Anything-2.1-RKNN2/resolve/main/sam2.1_hiera_small_encoder.rknn'

# SAM 2.1 Small — segmentation decoder (16 MB, runs on CPU)
wget -O ~/models/sam2.1/sam2.1_hiera_small_decoder.onnx \
  'https://huggingface.co/happyme531/Segment-Anything-2.1-RKNN2/resolve/main/sam2.1_hiera_small_decoder.onnx'
```

### Python dependencies for SAM

```bash
pip3 install --break-system-packages onnxruntime pillow matplotlib opencv-python-headless
```

## Step 7: Test Models (CLI)

### Text LLM
```bash
printf "What is the capital of France?\nexit\n" | \
  ~/rknn-llm/examples/rkllm_api_demo/deploy/build/llm_demo \
  ~/models/Qwen3-1.7B-w8a8-rk3588.rkllm 512 2048
```

Expected output includes: `The capital of France is Paris.`

### VLM (Vision)
```bash
printf "0\nexit\n" | \
  ~/rknn-llm/examples/multimodal_model_demo/deploy/build/demo \
  ~/rknn-llm/examples/multimodal_model_demo/data/demo.jpg \
  ~/models/qwen3-vl-2b_vision_448_rk3588.rknn \
  ~/models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm \
  256 2048 3
```

This analyzes the sample image (an astronaut on the moon) and describes it.

## Step 8: Deploy the Web App

The web app provides a browser-based UI with:
- **Chat** tab for text models (streaming responses)
- **Vision** tab for uploading images to the VLM
- **Live Stream** tab for real-time camera analysis via VLM
- Model switching (only one model loaded at a time to manage 16GB RAM)
- Memory monitoring and auto-protection against OOM crashes

### From your PC (Windows/Mac/Linux):

```bash
cd rock_llm/
bash deploy.sh
```

This copies all files to the board, builds the persistent VLM binary (`vlm_stream`), and installs Flask.

### Or manually on the board:

```bash
mkdir -p ~/rkllm-web
# Copy server.py, index.html, vlm_stream.cpp, CMakeLists.txt to ~/rkllm-web/

# Install Flask
pip3 install --break-system-packages flask

# Build the persistent VLM binary
cd ~/rkllm-web
mkdir -p build && cd build
cmake ..
make -j4
cp vlm_stream ~/rkllm-web/

# Copy supporting files
cp ~/rknn-llm/scripts/fix_freq_rk3588.sh ~/rkllm-web/
```

### Start the server:

```bash
cd ~/rkllm-web
python3 server.py
```

Open in your browser: `http://<board-ip>:8080`

For camera/live stream access (requires HTTPS): `https://<board-ip>:8081`
(Accept the self-signed certificate warning on first visit)

## Step 9: Set Up as a Service (Auto-Start on Boot)

```bash
sudo tee /etc/systemd/system/rkllm-web.service << 'EOF'
[Unit]
Description=RKLLM Web Server
After=network.target

[Service]
Type=simple
User=rock
WorkingDirectory=/home/rock/rkllm-web
ExecStart=/usr/bin/python3 /home/rock/rkllm-web/server.py
Restart=always
RestartSec=10
Environment=PATH=/home/rock/.local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rkllm-web
sudo systemctl start rkllm-web
```

Manage with:
```bash
sudo systemctl status rkllm-web    # check status
sudo systemctl restart rkllm-web   # restart
sudo systemctl stop rkllm-web      # stop
journalctl -u rkllm-web -f         # live logs
```

---

## System Tuning (Recommended)

### Disable Desktop Environment

The KDE desktop uses ~500MB RAM. Since you access the board via SSH and browser, disable it:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

To re-enable later: `sudo systemctl set-default graphical.target`

### Auto-Login (No Password Prompt on Console)

```bash
# For GDM (graphical login)
sudo tee /etc/gdm3/daemon.conf << 'EOF'
[daemon]
AutomaticLoginEnable=true
AutomaticLogin=rock
EOF

# For TTY (text console)
sudo mkdir -p /etc/systemd/system/getty@tty1.service.d
sudo tee /etc/systemd/system/getty@tty1.service.d/override.conf << 'EOF'
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin rock --noclear %I $TERM
EOF
```

### Passwordless Sudo

```bash
echo "rock ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/rock
sudo chmod 440 /etc/sudoers.d/rock
```

### Add Disk-Based Swap

The default zram swap compresses into RAM — useless under memory pressure. Add a real swap file:

```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo "/swapfile none swap sw,pri=50 0 0" | sudo tee -a /etc/fstab

# Reduce swappiness to avoid zram thrashing
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
sudo sysctl vm.swappiness=10
```

---

## Architecture & Memory Usage

```
┌─────────────────────────────────────────────────────┐
│                  16 GB RAM                          │
├───────────┬─────────────────────────────────────────┤
│ OS + Swap │ ~1.0 GB (headless) / ~1.5 GB (desktop) │
│ Qwen3 1.7B│ ~2.5 GB                                │
│ Qwen3 4B  │ ~5.5 GB                                │
│ Qwen3-VL  │ ~3.5 GB (LLM + vision encoder)         │
│ SAM 2.1   │ ~1.2 GB (encoder + decoder)             │
│ Free      │ Remainder (safety buffer)               │
└───────────┴─────────────────────────────────────────┘
```

Only **one model is loaded at a time**. The web server enforces this — switching models unloads the current one first.

### Memory Safety

The server includes a watchdog thread that:
- Monitors available RAM every 10 seconds
- Unloads the model if free RAM drops below 500 MB
- Restarts the VLM process every 200 inferences to reclaim memory leaked by the NPU runtime (closed-source Rockchip libraries)
- The browser auto-stops live stream after 5 consecutive errors

---

## File Structure

### On the board (`~/rkllm-web/`)

| File | Purpose |
|------|---------|
| `server.py` | Unified Flask server — text LLM (ctypes), VLM (subprocess), web UI |
| `index.html` | Single-page web UI with Chat, Vision, Live Stream tabs |
| `vlm_stream` | Compiled C++ binary — persistent VLM that keeps models loaded between inferences |
| `vlm_stream.cpp` | Source for the above |
| `CMakeLists.txt` | Build config |
| `fix_freq_rk3588.sh` | Locks NPU frequency for stable performance |
| `ssl/` | Auto-generated self-signed cert for HTTPS (camera access) |
| `start.sh` | Convenience script to start the server |
| `restart.sh` | Kill and restart |

### Models (`~/models/`)

| File | Size |
|------|------|
| `Qwen3-1.7B-w8a8-rk3588.rkllm` | 2.3 GB |
| `Qwen3-4B-w8a8-rk3588.rkllm` | 5.0 GB |
| `qwen3-vl-2b-instruct_w8a8_rk3588.rkllm` | 2.3 GB |
| `qwen3-vl-2b_vision_448_rk3588.rknn` | 850 MB |
| `sam2.1/sam2.1_hiera_small_encoder.rknn` | 358 MB |
| `sam2.1/sam2.1_hiera_small_decoder.onnx` | 16 MB |

### On your PC (`rock_llm/`)

Edit files here, then deploy with `bash deploy.sh`. The script SCPs everything to the board and rebuilds.

---

## API Reference

The server exposes an HTTP API at port 8080:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/status` | GET | Current model, memory usage, busy state |
| `/api/models` | GET | List available models |
| `/api/load` | POST | Load a model: `{"model": "qwen3-1.7b"}` |
| `/api/unload` | POST | Unload current model |
| `/api/chat` | POST | Text chat: `{"messages": [{"role":"user","content":"..."}], "stream": true}` |
| `/api/vlm` | POST | VLM: `{"image": "<base64>", "prompt": "..."}` |
| `/api/vlm/stream` | POST | Live stream VLM (skips if busy) |
| `/api/stop` | POST | Stop current LLM generation |
| `/api/sam/encode` | POST | Encode image for SAM: `{"image": "<base64>"}` |
| `/api/sam/segment` | POST | Segment with points: `{"points": [[x,y]], "labels": [1]}` |
| `/api/sam/auto` | POST | Auto-segment frame (for live stream) |
| `/rkllm_chat` | POST | Legacy RKLLM API compatibility |

### Example: Text Chat

```bash
curl -X POST http://192.168.1.41:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":false}'
```

### Example: Python Client

```python
import requests

# Load a model
requests.post("http://192.168.1.41:8080/api/load", json={"model": "qwen3-1.7b"})

# Chat
r = requests.post("http://192.168.1.41:8080/api/chat", json={
    "messages": [{"role": "user", "content": "What is gravity?"}],
    "stream": False
})
print(r.json()["choices"][0]["message"]["content"])
```

---

## Troubleshooting

### "matmul(w8a8) run failed"
RKNPU driver is too old. You need >= 0.9.7. Upgrade the kernel (Step 2).

### Board freezes / solid green LED
OOM crash. The NPU runtime leaks memory over long sessions. Ensure the watchdog is active in `server.py` and the desktop environment is disabled.

### Camera doesn't work in browser
Browsers require HTTPS for `getUserMedia`. Use `https://<board-ip>:8081` and accept the self-signed certificate warning.

### VLM returns 500 errors
The `vlm_stream` process may have died. Check with:
```bash
journalctl -u rkllm-web -n 50
```
Restart the service: `sudo systemctl restart rkllm-web`

### Model fails to load
Not enough RAM. Unload the current model first, or switch to a smaller one. Check with:
```bash
free -h
```

---

## Key Versions

| Component | Version |
|-----------|---------|
| OS | Debian 12 (Bookworm) |
| Kernel | 6.1.84-8-rk2410 |
| RKNPU Driver | 0.9.8 |
| RKNPU2 Userspace | 2.3.0 |
| RKLLM Runtime | 1.2.3 |
| RKLLM Toolkit | 1.2.3 |
| Python | 3.11 |
| Flask | 3.1 |

---

## Additional Models to Try

Pre-converted RKLLM models for RK3588 can be found on HuggingFace. Search for repos tagged with `rkllm` or `rk3588`:

- **Text:** Gemma 3 4B, Phi 3 Mini, DeepSeek
- **Vision:** Qwen3-VL 4B, Qwen2-VL, MiniCPM-V-2.6, InternVL

Models must be in `.rkllm` format (converted with `rkllm-toolkit`). The W8A8 quantization works best on RK3588.
