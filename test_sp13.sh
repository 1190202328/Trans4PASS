#!/bin/bash

file=tools/train_sp.py
cfg=configs/synpass13/trans4pass_plus_small_512x512.yaml

## eval oracle
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  $file --test --config-file $cfg

## eval source only single source
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  tools/eval_dp13.py --config-file $cfg

## [usually not to do]eval domain adaptation
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  adaptations/evaluate_out13.py
