#!/bin/bash

# step 0 train on source
# Trans4PASS
cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
  tools/train_cs.py --config-file configs/cityscapes/trans4pass_tiny_512x512.yaml
