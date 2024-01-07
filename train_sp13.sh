#!/bin/bash

# step 0 train on source
# Trans4PASS
cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
  tools/train_sp.py --config-file configs/synpass13/trans4pass_plus_small_512x512.yaml
