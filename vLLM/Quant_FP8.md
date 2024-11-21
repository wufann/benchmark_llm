# 1 download quark package
```
https://xcoartifactory/artifactory/uai-pip-local/com/amd/quark/main/nightly/
quark-1.0.0.dev20241019+e0b5262c50.zip
unzip quark-1.0.0.dev20241019+e0b5262c50.zip
```
# 2 install quark
```
pip install quark-1.0.0.dev20241019+e0b5262c50-py3-none-any.whl
```
# 3 quant model from origin data type to fp8 dtype
```
cd examples/torch/language_modeling/llm_ptq/
python3 quantize_quark.py --model_dir /path/to/llama-2-70b-chat-hf/ \
                         --output_dir /path/to/Llama-2-70b-FP8-quark \
                         --quant_scheme w_fp8_a_fp8 \
                         --kv_cache_dtype fp8 \
                         --num_calib_data 128 \
                         --model_export quark_safetensors
```