#!/usr/bin/env bash

RUN_VERSION='1_aug_gray_1_5_1_stage_2_v1'

#python2 tools/test_net.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-C4_1x.yaml \
#OUTPUT_DIR detectron/datasets/data/results/${RUN_VERSION}/

python2 detectron/datasets/nuclei/write_submission.py \
    --results-root detectron/datasets/data/ \
    --run-version ${RUN_VERSION} \
    --iters '65999' \
    --area-thresh 15 \
    --acc-thresh 0.35 \
    --intersection-thresh 0.3