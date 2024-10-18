# Launch the server
# Meta-Llama-3-70B-Instruct-FP8
python -m sglang.launch_server --model-path neuralmagic/Meta-Llama-3-70B-Instruct-FP8 --disable-radix-cache --tp 8 --port 10086
 
# Offline benchmark
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 2048 --output-file offline.jsonl --port 10086

# # Benchmark Result
# ============ Serving Benchmark Result ============
# Backend:                                 sglang
# Traffic request rate:                    inf
# Successful requests:                     1000
# Benchmark duration (s):                  317.30
# Total input tokens:                      2033482
# Total generated tokens:                  1027062
# Total generated tokens (retokenized):    1014917
# Request throughput (req/s):              3.15
# Input token throughput (tok/s):          6408.67
# Output token throughput (tok/s):         3236.86
# ----------------End-to-End Latency----------------
# Mean E2E Latency (ms):                   209386.65
# Median E2E Latency (ms):                 229738.62
# ---------------Time to First Token----------------
# Mean TTFT (ms):                          81468.70
# Median TTFT (ms):                        42949.64
# P99 TTFT (ms):                           232866.49
# -----Time per Output Token (excl. 1st token)------
# Mean TPOT (ms):                          181.91
# Median TPOT (ms):                        133.67
# P99 TPOT (ms):                           738.82
# ---------------Inter-token Latency----------------
# Mean ITL (ms):                           128.60
# Median ITL (ms):                         99.68
# P99 ITL (ms):                            418.23
# ==================================================

# Launch vLLM server
python3 -m vllm.entrypoints.openai.api_server --model /scratch/workspace/fanwu103/models/Meta-Llama-3-70B-Instruct-FP8  --disable-log-requests --tensor 8
 
# vLLM Offline benchmark
wget https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/bench_serving.py
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 2048 --output-file offline_vllm.jsonl
 
# # vLLM Benchmark Result
# ============ Serving Benchmark Result ============
# Backend:                                 vllm
# Traffic request rate:                    inf
# Successful requests:                     997
# Benchmark duration (s):                  422.29
# Total input tokens:                      2024829
# Total generated tokens:                  1027062
# Total generated tokens (retokenized):    1023829
# Request throughput (req/s):              2.36
# Input token throughput (tok/s):          4794.89
# Output token throughput (tok/s):         2432.13
# ----------------End-to-End Latency----------------
# Mean E2E Latency (ms):                   246100.50
# Median E2E Latency (ms):                 251221.21
# ---------------Time to First Token----------------
# Mean TTFT (ms):                          151387.69
# Median TTFT (ms):                        154039.37
# P99 TTFT (ms):                           342580.10
# -----Time per Output Token (excl. 1st token)------
# Mean TPOT (ms):                          108.52
# Median TPOT (ms):                        97.00
# P99 TPOT (ms):                           272.84
# ---------------Inter-token Latency----------------
# Mean ITL (ms):                           92.42
# Median ITL (ms):                         89.97
# P99 ITL (ms):                            195.94
# ==================================================
