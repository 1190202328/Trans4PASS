#!/bin/bash

DIR=.

# TODO 需要解开
#bash /nfs/volume-902-16/tangwenbo/ofs-1.sh

# step1 warm-up
cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
  train_warm.py
# step2 generate pseudo labels
cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
  gen_pseudo_label.py

# step3 domain adaptation
# (optional) python train_ssl.py
cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
  train_mpa.py
