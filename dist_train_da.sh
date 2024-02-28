#!/bin/bash

bash /nfs/volume-902-16/tangwenbo/s3_all.sh

exp_name=$1

# step1 warm-up
cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
  train_warm.py --snapshot-dir="$exp_name"

## step1.5 generate pseudo labels
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  gen_pseudo_label.py

## step2 ssl learning
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  train_ssl.py

## step2.5 generate pseudo labels
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  gen_pseudo_label.py

## step3 domain adaptation
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  train_mpa_out_p2p.py
