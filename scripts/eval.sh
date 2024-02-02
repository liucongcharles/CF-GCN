#!/usr/bin/env bash

gpus=1

data_name=LEVIR
net_G=base_GCN
split=test
project_name=BIT_LEVIR
checkpoint_name=/root/lc/changeGCN_GPU2/BIT_CD-master/checkpoints/best_ckpt.pt

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


