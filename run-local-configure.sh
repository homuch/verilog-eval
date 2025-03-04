#!/bin/bash

SCRIPT_REALPATH=$(realpath "$0")
SCRIPT_DIR_REAL=$(dirname "$SCRIPT_REALPATH")

source $SCRIPT_DIR_REAL/configurations.sh

mkdir -p $SCRIPT_DIR_REAL/build

cd $SCRIPT_DIR_REAL/build

env PATH="$IVERILOG_PATH:$PATH" ../configure \
    --with-task=$task \
    --with-model=$model \
    --with-examples=$shots \
    --with-samples=$samples \
    --with-temperature=$temperature \
    --with-top-p=$top_p
