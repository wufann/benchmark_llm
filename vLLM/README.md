# How to Tune OP 
## 1. Tune GEMM
### 1.1 use PyTorch TunableOp
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

### 1.2 use Gradlib
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

## 2. Tune Fused_moe
```
https://github.com/ROCm/vllm/pull/143/
install vllm: PYTORCH_ROCM_ARCH=gfx942 python3 setup.py develop
cd benchmarks/kernels
torchrun benchmark_mixtral_moe_rocm.py --model 8x7B --modelTP 8 --numGPU 8 --use_fp8
tune_file: 'E=8,N=2048,device_name=AMD_Instinct_MI300X_OAM,dtype=fp8_w8a8.json', copy to /model_excutor/layers/fused_model/config/
```