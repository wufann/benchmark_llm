set -xe
model_name=llama2-7b
tp=1
precision=w8a8sq
huggingface_dir=/data/models/Llama-2-7b-hf
trt_ckpt_dir=`pwd`/models/ckpt
example_dir=/app/tensorrt_llm/examples/llama
echo "====== convert ckpt from ${huggingface_dir} to ${trt_ckpt_dir} ======"
if [ ! -f "${trt_ckpt_dir}/${model_name}_${precision}_tp${tp}/rank0.safetensors" ]; then
    python3 ${example_dir}/convert_checkpoint.py \
        --model_dir ${huggingface_dir}/ \
        --output_dir ${trt_ckpt_dir}/${model_name}_${precision}_tp${tp} \
        --dtype float16 \
        --tp_size ${tp} \
	--per_channel \
	--per_token \
	--smoothquant 0.5
fi

engine_dir=`pwd`/models/engines
max_batch_size=128
max_input_len=2048
max_output_len=128
max_seq_len=$((max_input_len + max_output_len))
max_num_tokens=$((max_batch_size * max_input_len))
# ref: https://nvidia.github.io/TensorRT-LLM/performance/perf-best-practices.html#build-options-to-optimize-the-performance-of-tensorrt-llm-models
trtllm-build \
    --checkpoint_dir ${trt_ckpt_dir}/${model_name}_${precision}_tp${tp} \
    --output_dir ${engine_dir}/${model_name}_${precision}_tp${tp} \
    --max_batch_size ${max_batch_size} \
    --max_input_len ${max_input_len} \
    --max_seq_len ${max_seq_len} \
    --max_num_tokens ${max_num_tokens} \
    --workers ${tp} \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16
