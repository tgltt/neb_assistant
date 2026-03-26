#!/bin/bash
pip install peft
# pip install llmuses

# cd vllm-main
# pip install pyproject
# pip install -e .
# cd ..

pip install chromadb
pip install langchain langchain_core langchain_community langchain_openai langgraph langchain_chroma

# cd LLaMA-Factory-main

# pip install -r requirements.txt
# pip install -e ".[torch,metrics]"

# pip install git+https://github.com/huggingface/transformers accelerate
# pip install qwen-vl-utils

# cd ..

pip install -U vllm

pip install streamlit
pip install pymupdf

cp -v assets/frpc_linux_amd64 /root/miniconda3/lib/python3.11/site-packages/gradio/frpc_linux_amd64_v0.2
chmod +x /root/miniconda3/lib/python3.11/site-packages/gradio/frpc_linux_amd64_v0.2
