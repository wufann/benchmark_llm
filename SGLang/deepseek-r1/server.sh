# launch server
python3 -m sglang.launch_server --model /path/to/DeepSeek-R1 --trust-remote-code --tp 8

## best performence
#python3 -m sglang.launch_server --model /path/to/DeepSeek-R1 --trust-remote-code --tp 8 --chunked-prefill-size 130172 --enable-torch-compile