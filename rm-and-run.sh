#!/usr/bin/env bash

docker stop inferrer
docker rm inferrer
nvidia-docker build -t detectron .
nvidia-docker run -it --name inferrer detectron /bin/bash

#./bin/nuclei/generate_submission.sh
docker start inferrer
docker cp inferrer:/detectron/test2.csv test2.csv