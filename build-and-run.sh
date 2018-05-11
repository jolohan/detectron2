#!/bin/bash

docker build -t detectron .
echo "docker built"
if [ -z "$1" ]
  then
    echo "No argument supplied"
else
    nvidia-docker run -it --name yellow_submarine detectron $1
    nvidia-docker stop yellow_submarine
    nvidia-docker rm yellow_submarine
fi