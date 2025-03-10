#!/bin/bash

SCRIPT_REALPATH=$(realpath "$0")
SCRIPT_DIR_REAL=$(dirname "$SCRIPT_REALPATH")

source $SCRIPT_DIR_REAL/configurations.sh

echo "Running local-transformer.sh with:"
echo "task: $task"
echo "model: $model"
echo "temperature: $temperature"
echo "top_p: $top_p"
echo "max_tokens: $max_tokens"
echo "samples: $samples"
echo "model_runner: $model_runner"
echo "remove_system_prompt: $remove_system_prompt"

mkdir -p $SCRIPT_DIR_REAL/build

python3 manual_run/manual-transformer-run.py \
    --task=$task \
    --model=$model \
    --temperature=$temperature \
    --top_p=$top_p \
    --max_tokens=$max_tokens \
    --samples=$samples \
    --remove_system_prompt=$remove_system_prompt \
    --model_runner=$model_runner
