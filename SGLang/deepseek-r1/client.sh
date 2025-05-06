# client
ISL=3200
OSL=800
CON="1 4 16 32 64 128"
PROMPTS=500
RATIO=1.0
for con in $CON; do
    python3 -m sglang.bench_serving \
        --dataset-name random \
        --random-input-len $ISL \
        --random-output-len $OSL \
        --num-prompt $PROMPTS \
        --random-range-ratio $RATIO \
        --max-concurrency $con
done