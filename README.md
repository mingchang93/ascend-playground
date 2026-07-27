```shell
docker run -u 0 -dit --shm-size=512g --name=lmc_flexattn --net=host --privileged \
--security-opt seccomp=unconfined \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /data:/data \
-v /data2:/data2 \
-v /data3:/data3 \
swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11 \
/bin/bash
```