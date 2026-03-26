#!/bin/bash
# python -m vllm.entrypoints.openai.api_server --model /gemini/code/neb_assistant/models/qwen/Qwen2-VL-7B-Instruct --host 0.0.0.0 --port 8000

# python -m vllm.entrypoints.openai.api_server --model /gemini/code/neb_assistant/saves/Qwen2VL-7B-Chat/lora/train_2024-09-12-01-20-48 --host 0.0.0.0 --port 8000
python -m vllm.entrypoints.openai.api_server --model /gemini/code/neb_assistant/models/qwen/Qwen2-VL-7B-Instruct-merged --host 0.0.0.0 --port 8000