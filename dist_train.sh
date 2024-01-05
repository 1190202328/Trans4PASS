#!/bin/bash

bash /nfs/volume-902-16/tangwenbo/s3_all.sh

# step 0 train on source
# Trans4PASS
cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS && CUDA_VISIBLE_DEVICES=0,1,2,3 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
  -m torch.distributed.launch --nproc_per_node=4 tools/train_cs.py --config-file configs/cityscapes/trans4pass_tiny_512x512.yaml

## step1 warm-up
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  train_warm.py
## step2 generate pseudo labels
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  gen_pseudo_label.py
#
## step3 domain adaptation
## (optional) python train_ssl.py
#cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/Trans4PASS/adaptations && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
#  train_mpa.py
