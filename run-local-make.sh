#!/bin/bash

SCRIPT_REALPATH=$(realpath "$0")
SCRIPT_DIR_REAL=$(dirname "$SCRIPT_REALPATH")

mkdir -p $SCRIPT_DIR_REAL/build

cd $SCRIPT_DIR_REAL/build

source $SCRIPT_DIR_REAL/configurations.sh

env PATH="$IVERILOG_PATH:$PATH" make -j$(nproc)
