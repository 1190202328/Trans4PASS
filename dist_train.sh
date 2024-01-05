#!/bin/bash

bash /nfs/volume-902-16/tangwenbo/s3_all.sh

file=tools/train_cs13.py
cfg=configs/cityscapes13/trans4pass_plus_small_512x512.yaml

# step 0 train on source
# Trans4PASS
cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS && CUDA_VISIBLE_DEVICES=0,1,2,3 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
  -m torch.distributed.launch --nproc_per_node=4 $1 --config-file $2
