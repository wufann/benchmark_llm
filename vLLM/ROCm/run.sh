#!/usr/bin/bash
export PYTORCH_TUNABLEOP_ENABLED=1
./vllm_benchmark_report.sh -s latency -m /models/Llama-2-7b-FP8-quark/ -g 1 -d float8 2>&1 | tee log_7b_fp8_tp1
./vllm_benchmark_report.sh -s latency -m /models/Llama-2-7b-FP8-quark/ -g 2 -d float8 2>&1 | tee log_7b_fp8_tp2

./vllm_benchmark_report.sh -s latency -m /models/Llama-2-7b-chat-hf -g 1 -d float16 2>&1 | tee log_7b_fp16_tp1
./vllm_benchmark_report.sh -s latency -m /models/Llama-2-7b-chat-hf -g 2 -d float16 2>&1 | tee log_7b_fp16_tp2