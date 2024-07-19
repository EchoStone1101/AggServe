#!/bin/bash

# 为参数设置默认值
NUM_PROMPTS_REQ_RATES="${1:-'[(3000, 20)]'}"
EXP_RESULT_DIR="${2:-mps-tp}"
BACKEND="${3:-distserve-mps}"
ADDITIONAL_ARG="${4:-default_value}"

python evaluation/2-benchmark-serving/2-benchmark-serving.py --dataset /opt/tiger/AggServe/dataset/sharegpt-fixedlen-1024-96.marshal --process-name uniform --exp-result-root ./exp --num-prompts-req-rates "$NUM_PROMPTS_REQ_RATES" --exp-result-dir "$EXP_RESULT_DIR" --backend "$BACKEND" "$ADDITIONAL_ARG"