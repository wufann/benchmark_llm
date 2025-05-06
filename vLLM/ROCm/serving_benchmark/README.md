# 1. download model：
llama4
qwen3

# 2. start container
```
#/bin/bash
export MY_CONTAINER="xxx_vllm"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
docker run -e  DISPLAY=$DISPLAY --net=host --pid=host --ipc=host \
        --shm-size 64g \
        --privileged \
        -it \
        -v /tools/:/tools/ \
        -v /path/to/models/:/path/to/models/ \
        --name $MY_CONTAINER  \
        xxxx_docker_image \
        /bin/bash
else
docker start $MY_CONTAINER
docker exec -ti $MY_CONTAINER /bin/bash
fi
```

# 3. launch server
```
# set real model path in server.sh
bash server.sh
```

# 4. client
```
bash client.sh
```