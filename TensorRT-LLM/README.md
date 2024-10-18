# Building a TensorRT-LLM docker image
ref: https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#option-1-build-tensorrt-llm-in-one-step
```
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs
git lfs install
 
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull
git checkout （release version）
 
# (optional)Modify the docker/Makefile IMAGE_TAG
-IMAGE_TAG          ?= latest
+IMAGE_TAG          ?= cuda-12.5-trtllm-v0.13.0-fanwu103
 
# build docker image
make -C docker release_build CUDA_ARCHS="90-real" // 90-real is Nvidia Hopper architecture

# check if docker images has build
docker ps -a | grep tensorrt_llm/release:cuda-12.5-trtllm-v0.13.0-fanwu103
```

# Starting container based on builded images
```
#/bin/bash
export MY_CONTAINER="trtllm-v0.13-fanwu103"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
docker run -e DISPLAY=$DISPLAY --net=host --pid=host --ipc=host \
        --shm-size 64g \
        --privileged \
        -it \
        --gpus all \
        -v /tools/:/tools/ \
        -v /scratch/:/scratch/ \
        -v /data/:/data/ \
        -v /home/fanwu103/:/home/fanwu103/ \
        --name $MY_CONTAINER  \
        tensorrt_llm/release:cuda-12.5-trtllm-v0.13.0-fanwu103 \
        /bin/bash
else
docker start $MY_CONTAINER
docker exec -ti $MY_CONTAINER /bin/bash
fi
```
