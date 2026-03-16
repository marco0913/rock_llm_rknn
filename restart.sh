#!/bin/bash
pkill -f "python3 server.py" 2>/dev/null
sleep 2
cd ~/rkllm-web
export PATH="$HOME/.local/bin:$PATH"
nohup python3 server.py > /tmp/rkllm-web.log 2>&1 &
echo "Server started (PID: $!)"
sleep 3
tail -5 /tmp/rkllm-web.log
