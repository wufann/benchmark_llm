# generate dataset
python /app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py --output=norm-dist-i2048-o128-n2000.json \
    --tokenizer=/data/models/Llama-2-7b-hf/ \
    token-norm-dist \
    --num-requests 2000 \
    --input-mean=2048 \
    --output-mean=128 \
    --input-stdev=0 \
    --output-stdev=0

# run gptManagerBenchmark
mpirun --allow-run-as-root -n 1 /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark \
    --engine_dir models/engines/llama2-7b_fp16_tp1/ \
    --type IFB --request_rate -1 \
    --static_emulated_batch_size 128 \
    --max_num_samples 128 \
    --enable_kv_cache_reuse false --streaming true \
    --dataset norm-dist-i2048-o128-n2000.json \
    --output_csv output-i2048-o128-n2000.csv
