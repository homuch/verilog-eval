#!/bin/bash

SCRIPT_REALPATH=$(realpath "$0")
SCRIPT_DIR_REAL=$(dirname "$SCRIPT_REALPATH")

task="code-complete-iccad2023"
model="manual-rtl-coder"
# model_path="/llm-models/RTLCoder-Deepseek-v1.1.Q4_K_S.gguf"
shots=0 # 0~3
samples=1
temperature=0.8
top_p=0.95

mkdir -p $SCRIPT_DIR_REAL/build

podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable \
    --volume $SCRIPT_DIR_REAL/../llm-models:/llm-models \
    --volume $SCRIPT_DIR_REAL/build:/build \
    llamacpp-build \
    python3 manual-run.py \
        --task=$task \
        --model=$model \
        --temperature=$temperature \
        --top_p=$top_p
