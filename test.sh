#!/bin/bash

DIR=.

#bash /nfs/volume-902-16/tangwenbo/ofs-1.sh

# eval single_source
cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
  tools/eval_dp.py --config-file configs/cityscapes/trans4pass_tiny_512x512.yaml

## eval domain adaptation
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  adaptations/evaluate.py
