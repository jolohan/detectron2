#!/usr/bin/env bash

docker stop test_demo
docker rm test_demo
nvidia-docker build -t detectron .
nvidia-docker run -it --name test_demo detectron /bin/bash

#./bin/nuclei/generate_submission.sh
docker start test_demo
#docker cp test_demo:/tmp/detectron-visualizations detectron_visualizations

#python2 tools/infer_simple.py \
#--cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-C4_1x.yaml \
#--output-dir /tmp/detectron-visualizations \
#--image-ext png \
#--wts configs/models/R-50.pkl \
#demo