#!/usr/bin/env bash

set -eu

gpu_id="1"

# CUDA_VISIBLE_DEVICES=$gpu_id python -u main_test.py 
# CUDA_VISIBLE_DEVICES=$gpu_id python -u test_attention_visual.py 
CUDA_VISIBLE_DEVICES=$gpu_id python -u main_test_avg.py
