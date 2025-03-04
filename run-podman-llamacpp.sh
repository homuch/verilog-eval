#!/bin/bash

SCRIPT_REALPATH=$(realpath "$0")
SCRIPT_DIR_REAL=$(dirname "$SCRIPT_REALPATH")

source $SCRIPT_DIR_REAL/configurations.sh

echo "Running podman-llamacpp.sh with:"
echo "task: $task"
echo "model: $model"
echo "temperature: $temperature"
echo "top_p: $top_p"
echo "max_tokens: $max_tokens"
echo "samples: $samples"

mkdir -p $SCRIPT_DIR_REAL/build

podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable \
    --volume $SCRIPT_DIR_REAL/../llm-models:/llm-models \
    --volume $SCRIPT_DIR_REAL/build:/build \
    llamacpp-build \
    python3 manual-run.py \
        --task=$task \
        --model=$model \
        --temperature=$temperature \
        --top_p=$top_p \
        --max_tokens=$max_tokens \
        --samples=$samples
