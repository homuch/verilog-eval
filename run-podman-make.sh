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
    make --jobs=1 all
    # python3 /app/build/../scripts/sv-generate \
    # "--model=manual-rtl-coder" "--examples=0" "--task=code-complete-iccad2023" \
    # "--temperature=0.8" "--top-p=0.95" \
    # --verbose \
    # --output Prob154_fsm_ps2data/Prob154_fsm_ps2data_sample01.sv \
    # /app/build/../dataset_code-complete-iccad2023/Prob154_fsm_ps2data_prompt.txt
