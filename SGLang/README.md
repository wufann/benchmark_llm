# Hardware
CPU: AMD EPYC 9654 96-Core Processor, Sockets = 4, Cores per socket = 96.
GPU: 8 * NVIDIA H100 80GB HBM3
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV18    NV18    NV18    NV18    NV18    NV18    NV18    0-95,192-287    0               N/A
GPU1    NV18     X      NV18    NV18    NV18    NV18    NV18    NV18    0-95,192-287    0               N/A
GPU2    NV18    NV18     X      NV18    NV18    NV18    NV18    NV18    0-95,192-287    0               N/A
GPU3    NV18    NV18    NV18     X      NV18    NV18    NV18    NV18    0-95,192-287    0               N/A
GPU4    NV18    NV18    NV18    NV18     X      NV18    NV18    NV18    96-191,288-383  1               N/A
GPU5    NV18    NV18    NV18    NV18    NV18     X      NV18    NV18    96-191,288-383  1               N/A
GPU6    NV18    NV18    NV18    NV18    NV18    NV18     X      NV18    96-191,288-383  1               N/A
GPU7    NV18    NV18    NV18    NV18    NV18    NV18    NV18     X      96-191,288-383  1               N/A
 
# Software
docker image:lmsysorg/sglang:v0.2.14-cu124
torch: 2.4.0
sglang: 0.2.14
vllm: 0.5.5
flashinfer: 0.1.5+cu124torch2.4
vllm-flash-attn: 2.6.1

# Models 
LLM models:Llama-3-70B (bf16 and fp8)
https://huggingface.co/neuralmagic/Meta-Llama-3-70B-Instruct-FP8
https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct

# How to reproduce the benchmark results of SGLang and vLLM
ref: https://github.com/sgl-project/sglang/blob/v0.2.14/benchmark/blog_v0_2/README.md
# Launch the server
# Meta-Llama-3-70B-Instruct-FP8
python -m sglang.launch_server --model-path neuralmagic/Meta-Llama-3-70B-Instruct-FP8 --disable-radix-cache --tp 8 --port 10086
 
# Offline benchmark
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 2048 --output-file offline.jsonl --port 10086
 
# Benchmark Result
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Successful requests:                     1000
Benchmark duration (s):                  317.30
Total input tokens:                      2033482
Total generated tokens:                  1027062
Total generated tokens (retokenized):    1014917
Request throughput (req/s):              3.15
Input token throughput (tok/s):          6408.67
Output token throughput (tok/s):         3236.86
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   209386.65
Median E2E Latency (ms):                 229738.62
---------------Time to First Token----------------
Mean TTFT (ms):                          81468.70
Median TTFT (ms):                        42949.64
P99 TTFT (ms):                           232866.49
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          181.91
Median TPOT (ms):                        133.67
P99 TPOT (ms):                           738.82
---------------Inter-token Latency----------------
Mean ITL (ms):                           128.60
Median ITL (ms):                         99.68
P99 ITL (ms):                            418.23
==================================================
 
# Launch vLLM server
python3 -m vllm.entrypoints.openai.api_server --model /scratch/workspace/fanwu103/models/Meta-Llama-3-70B-Instruct-FP8  --disable-log-requests --tensor 8
 
# vLLM Offline benchmark
wget https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/bench_serving.py
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 2048 --output-file offline_vllm.jsonl
 
# vLLM Benchmark Result
============ Serving Benchmark Result ============
Backend:                                 vllm
Traffic request rate:                    inf
Successful requests:                     997
Benchmark duration (s):                  422.29
Total input tokens:                      2024829
Total generated tokens:                  1027062
Total generated tokens (retokenized):    1023829
Request throughput (req/s):              2.36
Input token throughput (tok/s):          4794.89
Output token throughput (tok/s):         2432.13
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   246100.50
Median E2E Latency (ms):                 251221.21
---------------Time to First Token----------------
Mean TTFT (ms):                          151387.69
Median TTFT (ms):                        154039.37
P99 TTFT (ms):                           342580.10
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          108.52
Median TPOT (ms):                        97.00
P99 TPOT (ms):                           272.84
---------------Inter-token Latency----------------
Mean ITL (ms):                           92.42
Median ITL (ms):                         89.97
P99 ITL (ms):                            195.94
==================================================
