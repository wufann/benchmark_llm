# launch vLLM v0 server
unset VLLM_MOE_PADDING
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_USE_ROCM_FP8_FLASH_ATTN=1
export VLLM_ROCM_USE_AITER=0
# enable AITER is best
#export VLLM_ROCM_USE_AITER=1
vllm serve /path/to/model --enable-chunked-prefill False --trust-remote-code -tp 8 --swap-space 16 --disable-log-requests --dtype float16 --distributed-executor-backend mp