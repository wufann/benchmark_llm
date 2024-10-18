mpirun --allow-run-as-root -n 1 /app/tensorrt_llm/benchmarks/cpp/gptSessionBenchmark \
    --engine_dir models/engines/llama2-7b_fp16_tp1/ \
    --warm_up 2 \
    --num_runs 10 \
    --batch_size 64 \
    --input_output_len 2048,128
