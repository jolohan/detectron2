# Use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

RUN mv /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

# Update packages
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install numpy>=1.13 pyyaml>=3.12 matplotlib opencv-python>=3.2 setuptools Cython mock scipy easydict
RUN pip install pathlib joblib imageio xmltodict unicodecsv scipy

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Add remote directory
ADD ./ /detectron/

# Or clone git repo
#RUN git clone https://github.com/jolohan/detectron.git /Detectron


# install cython
RUN sudo pip install cython scikit-image easydict
# build cython extension
WORKDIR /detectron
RUN make

# Get to "home"
#WORKDIR ..

# Run inference
#RUN python2 tools/test_net.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml NUM_GPUS 1