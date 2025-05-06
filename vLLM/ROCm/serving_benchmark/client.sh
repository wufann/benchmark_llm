# client
ISL=3200
OSL=800
CON="1 4 16 32 64 128"
PROMPTS=500
RATIO=1.0
for con in $CON; do
    python3 benchmark_serving.py \
    --backend vllm \
    --model  /path/to/model \
    --dataset-name random \
    --max-concurrency $con \
    --random-input-len $ISL \
    --random-output-len $OSL \
    --random-range-ratio $RATIO \
    --num-prompts $PROMPTS
done