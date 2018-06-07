#!/usr/bin/env bash

set -e

echo "Location for models: "
echo $1
echo ".csv results file: " $2

cd ..

for filename in $1/*.pkl; do

    config='e2e_mask_rcnn_X-101-64x4d-FPN_1x_v2'

    python2 tools/test_net.py \
    --cfg configs/12_2017_baselines/$config.yaml \
    OUTPUT_DIR /detectron/output \
    TEST.WEIGHTS "$filename"

    base_filename=$(basename $filename)
    base_filename=${base_filename%.pkl}
    echo $base_filename

    cp '/detectron/output/test/dsb18_stage1_test/generalized_rcnn/segmentations_dsb18_stage1_test_results.json'

    python2 write_submission_from_json.py \
     '/detectron/output/test/dsb18_stage1_test/generalized_rcnn/segmentations_dsb18_stage1_test_results.json' \
     'detectron/datasets/data/dsb18/annotations/stage1_test.json' \
     '/detectron/output/results_/'$config'___'$base_filename

    python evaluate.py 'output/results_/' $config $base_filename

done