# FROM nvcr.io/nvidia/pytorch:19.12-py3
# FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel
# FROM nvcr.io/nvidia/tensorrt:20.02-py3
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
# FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive
ADD requirements.txt .
RUN apt-get update && \
    apt-get install -y git gitk curl python3 python3-pip
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#     python3 get-pip.py --force-reinstall && \
RUN pip3 install -U pip && \
    pip3 install --timeout 600 -r requirements.txt

# RUN pip install torch==1.3.0 torchvision==0.4.1
# RUN pip3 install torch==1.4.0 torchvision==0.5.0
# RUN pip3 install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# spconv
RUN pip3 install cmake && \
    apt-get install -y libboost-all-dev
RUN git clone https://github.com/traveller59/spconv.git --recursive && \
    cd spconv && \
    # git checkout 8da6f96 && \
    git submodule update --init --recursive && \
    python3 setup.py bdist_wheel && \
    cd dist && \
    pip3 install *.whl

# apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# nuscenes-devkit
RUN pip3 install setuptools==39.1.0 && \
    git clone https://github.com/poodarchu/nuscenes.git && \
    cd nuscenes && \
    python3 setup.py install

# install Det3D

# install tensorrt "7" for compatibility. may need to start from cuda 10.0?
# RUN os="ubuntu1604" && \
#     cuda="10.1.243" && \
#     wget https://developer.download.nvidia.com/compute/cuda/repos/${os}/x86_64/cuda-repo-${os}_${cuda}-1_amd64.deb && \
#     dpkg -i cuda-repo-*.deb
# RUN apt-get install -y libnvinfer7=7.0.0-1+cuda10.0 \
#                        libnvonnxparsers7=7.0.0-1+cuda10.0 \
#                        libnvparsers7=7.0.0-1+cuda10.0 \
#                        libnvinfer-plugin7=7.0.0-1+cuda10.0 \
#                        cuda-nvrtc-10-1 \
#                        libnvinfer-dev=7.0.0-1+cuda10.0 \
#                        libnvonnxparsers-dev=7.0.0-1+cuda10.0 \
#                        libnvparsers-dev=7.0.0-1+cuda10.0 \
#                        libnvinfer-plugin-dev=7.0.0-1+cuda10.0 \
#                        python-libnvinfer=7.0.0-1+cuda10.0 \
#                        python3-libnvinfer=7.0.0-1+cuda10.0

# onnx runtime-gpu
RUN pip3 install onnxruntime-gpu


# torch2trt
# RUN git clone https://github.com/arke812/torch2trt && \
#     cd torch2trt && \
#     # python3 setup.py install
#     python3 setup.py build develop
# # or python setup.py install --plugins

RUN apt-get install -y sudo
ENV USER_NAME=docker
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USER_NAME}
RUN chmod u+s /usr/sbin/useradd \
    && chmod u+s /usr/sbin/groupadd

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
