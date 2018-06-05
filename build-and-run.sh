#!/bin/bash

docker build -t detectron .
echo "docker built"
if [ -z "$1" ]
  then
    echo "No argument supplied"
else
  if [ -z "$2" ]
    then
      echo "No argument #2 supplied"
      nvidia-docker run -it --name yellow_submarine detectron $1
      nvidia-docker stop yellow_submarine
      #nvidia-docker rm yellow_submarine
  else
    echo "Argument #2 is $2"
    nvidia-docker run -it --name $2 detectron $1
  fi
fi
