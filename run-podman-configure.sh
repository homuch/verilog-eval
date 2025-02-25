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
    --volume $SCRIPT_DIR_REAL/build:/app/build \
    verilog-eval \
    ../configure \
        --with-task=$task \
        --with-model=$model \
        --with-examples=$shots \
        --with-samples=$samples \
        --with-temperature=$temperature \
        --with-top-p=$top_p
