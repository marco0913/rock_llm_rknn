#!/bin/bash
echo "============================================"
echo "  RKLLM Models on ROCK 5B (RK3588 NPU)"
echo "============================================"
echo ""
echo "Text Models:"
for f in ~/models/*.rkllm; do
    [ -f "$f" ] && printf "  %-55s %s\n" "$(basename $f)" "$(du -h $f | cut -f1)"
done
echo ""
echo "Vision Models:"
for f in ~/models/*.rknn; do
    [ -f "$f" ] && printf "  %-55s %s\n" "$(basename $f)" "$(du -h $f | cut -f1)"
done
[ ! -f ~/models/*.rknn ] 2>/dev/null && echo "  (none yet)"
echo ""
echo "Disk: $(df -h / | tail -1 | awk '{print "Used " $3 " / Free " $4 " / Total " $2}')"
echo "RAM:  $(free -h | awk 'NR==2{print "Used " $3 " / Free " $4 " / Total " $2}')"
echo ""
echo "Commands:"
echo "  ~/run-llm.sh qwen3-1.7b        # Fast, lightweight chat"
echo "  ~/run-llm.sh qwen3-4b           # Smarter, with thinking"
echo "  ~/run-vlm.sh image.jpg           # Vision + language"
echo "  ~/start-server.sh qwen3-1.7b    # HTTP API on port 8080"
echo "  ~/start-server.sh qwen3-4b      # HTTP API with 4B model"
