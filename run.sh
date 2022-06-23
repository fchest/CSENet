#!/usr/bin/env bash

set -eu

gpu_id="2"

CUDA_VISIBLE_DEVICES=$gpu_id python -u main.py
