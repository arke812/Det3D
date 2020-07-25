#!/bin/sh

if [ $# -eq 1 ]; then
TAG=det3d:$1
else
TAG=det3d:test
fi
KITTI_DATASET=/media/ken/EC-PHU3/dataset/kitti/object
OUTPUT_DIR=/home/ken/det3d_output

docker run --gpus all -it --rm \
           --shm-size=2g \
           -v `pwd`:/workspace/Det3D \
           -v ${KITTI_DATASET}:/dataset/kitti \
           -v ${OUTPUT_DIR}:/output \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -e DISPLAY=$DISPLAY \
           $TAG

# display settings in container
# export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0

# cd Det3D
# python3 setup.py build develop

# memo
# 
