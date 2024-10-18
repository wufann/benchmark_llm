wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_5/NsightSystems-linux-cli-public-2024.5.1.113-3461954.deb
dpkg -i https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_5/NsightSystems-linux-cli-public-2024.5.1.113-3461954.deb
 
## sglang
# launch server with Nsight Systems
nsys profile -o sglang_profile_1_prompt_d30_delay30_test_cudat --capture-range cudaProfilerApi -t cuda --force-overwrite true -d 30 --delay=30 python3 -m sglang.launch_server --model-path /scratch/huggingface_models/Meta-Llama-3.1-70B-Instruct-FP8/ --disable-radix-cache --tp 8 --port 10086 --chunked-prefill-size -1
# client request
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1 --random-input 4096 --random-output 2048 --output-file offline.jsonl --port 10086

## vllm
# launch server with Nsight Systems
nsys profile -o vllm_profile_1_prompt_d40_delay40_bf16 --capture-range cudaProfilerApi -t cuda --force-overwrite true -d 45 --delay=40 python3 -m vllm.entrypoints.openai.api_server --model  /scratch/huggingface_models/NousResearch-Meta-Llama-3.1-70B --disable-log-requests --tensor 8 --enable-chunked-prefill=False
# client request
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1 --random-input 4096 --random-output 2048 --output-file offline_vllm.jsonl
