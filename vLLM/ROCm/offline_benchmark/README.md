# scirpt copy form https://github.com/seungrokj/unified_docker_benchmark_public/tree/v2 

## Latency
```sh
# Usage: 
./vllm_benchmark_report.sh -s $mode -m $hf_model -g $n_gpu -d $datatype
# example:
# latency + throughput
./vllm_benchmark_report.sh -s all -m NousResearch/Meta-Llama-3-8B -g 1 -d float16
# latency 
./vllm_benchmark_report.sh -s latency -m NousResearch/Meta-Llama-3-8B -g 1 -d float16
# throughput
./vllm_benchmark_report.sh -s throughput -m NousResearch/Meta-Llama-3-8B -g 1 -d float16
```
