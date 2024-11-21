# How to Tune OP(GEMM and Fused_moe)
## 1. Tune GEMM
### 1.1 Offline fp16 GEMM tune based on **Gradlib**
install gradlib
```
git clone --recursive https://github.com/ROCm/vllm.git
cd vllm/gradlib
python setup.py install
```
generate GEMM M,N,K shape information in qwen2_fp16_untuned.csv
```
export VLLM_UNTUNE_FILE=./qwen2_fp16_untuned.csv
export VLLM_TUNE_GEMM=1
model_path=/datasets/Qwen2-7B/
python /app/vllm/benchmarks/benchmark_latency.py \
                --model ${model_path} \
                --trust-remote-code \
                --num-iters-warmup 0 \
                --num-iters 1 \
                --dtype float16 \
                --input-len 2048 \
                --output-len 128 \
                --batch-size ${bs} \
                --tensor-parallel-size ${tp_nums} \
                --num-scheduler-steps 10
```
tune gemm by gemm_tuner.py, generate qwen2_fp16_tuned.csv
```

python ../vllm/gradlib/gradlib/gemm_tuner.py --outdtype f16 --input_file qwen2_fp16_untuned.csv --tuned_file qwen2_fp16_tuned.csv
```
test tuned gemm performence:
```
export VLLM_TUNED_FILE=qwen2_fp16_tuned.csv
model_path=/datasets/Qwen2-7B/
python /app/vllm/benchmarks/benchmark_latency.py \
                --model ${model_path} \
                --trust-remote-code \
                --num-iters-warmup 0 \
                --num-iters 1 \
                --dtype float16 \
                --input-len 2048 \
                --output-len 128 \
                --batch-size ${bs} \
                --tensor-parallel-size ${tp_nums} \
                --num-scheduler-steps 10
```
### 1.2 Offline fp8 GEMM tune based on **afo**
install afo
```
git clone https://github.com/ROCm/pytorch_afo_testkit.git
cd pytorch_afo_testkit 
git checkout jpvillam/fp8_experimental_tun
pip install -e .
```
run case once，generate fp8 gemm shape
```
set -ex
export VLLM_USE_TRITON_FLASH_ATTN=0
model_path=/models/Llama-2-70b-FP8-quark
export HIPBLASLT_LOG_MASK=32
export HIPBLASLT_LOG_FILE=./hipblaslt-bench.log
for tp_nums in 8
    do
    for isl in 128 2048
    do
        for bs in 1 4 8 16 32 64 128
        do
            python /app/vllm/benchmarks/benchmark_latency.py \
                    --model ${model_path} \
                    --trust-remote-code \
                    --num-iters-warmup 0 \
                    --num-iters 1 \
                    --dtype float16 \
                    --kv-cache-dtype fp8 \
                    --input-len ${isl} \
                    --output-len 128 \
                    --batch-size ${bs} \
                    --tensor-parallel-size ${tp_nums} \
                    --num-scheduler-steps 16
        done
    done
done
```
get 'hipblaslt-bench.log'

tune fp8 gemm based afo
```
cat hipblaslt-bench.log | sort | uniq -c > hipblaslt-summary.log
python (/your/path/to/)pytorch_afo_testkit/afo/tools/tuning/hipblaslt2yaml.py hipblaslt-summary.log
afo tune result.yaml --cuda_device 0 1 2 3 4 5 6 7
```
get 'afo_tune_device_{device_id}_full.csv'

benchmark
```
export PYTORCH_TUNABLEOP_FILENAME=afo_tune_device_%d_full.csv
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_ENABLED=1
(your benchmark scipt):
加上上面三个环境变量，使用自己的benchmark脚本即可，例子：
export VLLM_USE_TRITON_FLASH_ATTN=0
model_path=/models/Llama-2-70b-FP8-quark
for tp_nums in 8
    do
    for isl in 128 2048
    do
        for bs in 1 4 8 16 32 64 128
        do
            python /app/vllm/benchmarks/benchmark_latency.py \
                    --model ${model_path} \
                    --trust-remote-code \
                    --num-iters-warmup 0 \
                    --num-iters 1 \
                    --dtype float16 \
                    --kv-cache-dtype fp8 \
                    --input-len ${isl} \
                    --output-len 128 \
                    --batch-size ${bs} \
                    --tensor-parallel-size ${tp_nums} \
                    --num-scheduler-steps 16
        done
    done
done
```

### 1.3 Online tuning based on **PyTorch TunableOp** (Not recommend)
```
First run for GEMM tunning
# Enable PyTorch TunableOp GEMM tuning
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_VERBOSE=0
export PYTORCH_TUNABLEOP_FILENAME=/dockerx/tunableop-config.csv
// Add tuning iteration
export PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=100 # default 30
export PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS=300 # default 100
   
Sec One for real measurement
# Enable PyTorch TunableOp (now using the tuned GEMMs)
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0 # highlight
export PYTORCH_TUNABLEOP_VERBOSE=0
export PYTORCH_TUNABLEOP_FILENAME=/dockerx/tunableop-config.csv
```


## 2. Tune Fused_moe
```
https://github.com/ROCm/vllm/pull/143/
install vllm: PYTORCH_ROCM_ARCH=gfx942 python3 setup.py develop
cd benchmarks/kernels
torchrun benchmark_mixtral_moe_rocm.py --model 8x7B --modelTP 8 --numGPU 8 --use_fp8
tune_file: 'E=8,N=2048,device_name=AMD_Instinct_MI300X_OAM,dtype=fp8_w8a8.json', copy to /model_excutor/layers/fused_model/config/
```