#!/bin/bash
# start_server.sh - Starts llama-server for the agent harness.
#
# Baseline configuration (project spec):
#   - 16k context (-c 16384), one slot (-np 1) so the whole 16k is available
#   - default build settings, 4 threads
#   - prompt caching is switched off by the CLIENT (cache_prompt=false in
#     harness/llm_client.py), as the spec requires for the baseline
#
# Usage:  bash scripts/start_server.sh [model.gguf]
#   default model: ~/models/qwen3-1.7b-bf16.gguf
# Stop with Ctrl+C. Check it is up: curl http://127.0.0.1:8080/health

MODEL="${1:-$HOME/models/qwen3-1.7b-bf16.gguf}"
exec "$HOME/llama.cpp/build/bin/llama-server" \
  -m "$MODEL" \
  -c 16384 \
  -np 1 \
  -t 4 \
  --host 127.0.0.1 \
  --port 8080
