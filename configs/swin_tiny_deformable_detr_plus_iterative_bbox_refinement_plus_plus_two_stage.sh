#!/usr/bin/env bash

set -x
# 训练地址
EXP_DIR=/root/autodl-tmp/DDM/exps/nwpu
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --num_queries_one2one 300 \
    --num_queries_one2many 0 \
    --k_one2many 0 \
    --epochs 36 \
    --lr_drop 20 \
    --dropout 0.0 \
    --mixed_selection \
    --look_forward_twice \
    --backbone swin_tiny \
    --pretrained_backbone_path /root/autodl-tmp/H-Deformable-DETR/configs/two_stage/deformable-detr-baseline/36eps/swin/swin_tiny_patch4_window7_224.pth \
    ${PY_ARGS}

