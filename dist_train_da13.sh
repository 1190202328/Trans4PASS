#!/bin/bash

bash /nfs/volume-902-16/tangwenbo/s3_all.sh

dataset_name=$1
exp_name=$2
start_step=$3
root_path="/nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/snapshots"
model_ckpt_name="best.pth"
pseudo_labels_dir_name="pseudo_labels"

warmup_global_path="${root_path}/my_${dataset_name}2DP13_Trans4PASS_plus_v2_WarmUp_${exp_name}"
ssl_global_path="${root_path}/my_${dataset_name}2DP13_Trans4PASS_plus_v2_SSL_${exp_name}"

if [ "${start_step}" -le 1 ]; then
  echo "step1: warm-up"
  cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
    train_warm_out_p2p.py --source="$dataset_name" --snapshot-dir="$exp_name"
fi

if [ "${start_step}" -le 2 ]; then
  echo "step2: generate pseudo labels for step1"
  cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
    gen_pseudo_label_out_p2p.py --restore-from="${warmup_global_path}/${model_ckpt_name}" --save="${warmup_global_path}/${pseudo_labels_dir_name}"
fi

if [ "${start_step}" -le 3 ]; then
  echo "step3: ssl learning"
  cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
    train_ssl_out_p2p.py --source="$dataset_name" --snapshot-dir="$exp_name" --restore-from="${warmup_global_path}/${model_ckpt_name}" --ssl-dir="${warmup_global_path}/${pseudo_labels_dir_name}"
fi

if [ "${start_step}" -le 4 ]; then
  echo "step4: generate pseudo labels for step3"
  cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
    gen_pseudo_label_out_p2p.py --restore-from="${ssl_global_path}/${model_ckpt_name}" --save="${ssl_global_path}/${pseudo_labels_dir_name}"
fi

#if [ "${start_step}" -le 5 ]; then
#  echo "step5: mpa"
#  cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#    train_mpa_out_p2p.py --source="$dataset_name" --snapshot-dir="$exp_name" --restore-from="${ssl_global_path}/${model_ckpt_name}" --ssl-dir="${ssl_global_path}/${pseudo_labels_dir_name}"
#fi
