#!/bin/bash

bash /nfs/volume-902-16/tangwenbo/s3_all.sh

exp_name=$1
start_step=$2
root_path="/nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/snapshots/"
step1_global_path="${root_path}/my_CS2DP_Trans4PASS_plus_v2_WarmUp_${exp_name}"

if [ "${start_step}" -le 1 ]; then
  echo "step1 warm-up"
  cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
    train_warm.py --snapshot-dir="$exp_name"
fi

if [ "${start_step}" -le 2 ]; then
  echo "step2 generate pseudo labels for step1"
  cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
    gen_pseudo_label.py --restore-from="${step1_global_path}/best.pth" --save="${step1_global_path}/pseudo_labels_warmup"
fi

if [ "${start_step}" -le 3 ]; then
  echo "step3 ssl learning"
fi

if [ "${start_step}" -le 4 ]; then
  echo "step4 generate pseudo labels for step3"
fi

if [ "${start_step}" -le 5 ]; then
  echo "step5 mpa"
fi

## step3 ssl learning
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  train_ssl.py

## step4 generate pseudo labels
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  gen_pseudo_label.py

## step5 domain adaptation
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  train_mpa_out_p2p.py
