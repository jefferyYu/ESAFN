#!/usr/bin/env bash
for d in 'twitter' #'twitter' 'twitter2015'
do
    for k in 'mmfusion' # 'mmram' 'mmmgan' 'mmfusion'
    do
        CUDA_VISIBLE_DEVICES=0 python train.py --dataset ${d} --embed_dim 100 --hidden_dim 100 --model_name ${k} \
        --att_mode vis_concat_attimg_gate
        # test --load_check_point >> ${d}_log.txt
    done
done
