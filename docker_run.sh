#!/bin/sh


docker run --gpus all -it --rm -v `pwd`:/workspace/Det3D \
           det3d:test

# display settings in container
# export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0

# cd Det3D
# python3 setup.py build develop
