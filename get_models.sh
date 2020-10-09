#!/bin/bash

wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz
tar xf faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz
rm faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz
tar xf efficientdet_d7_coco17_tpu-32.tar.gz
rm efficientdet_d7_coco17_tpu-32.tar.gz
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
tar xf efficientdet_d0_coco17_tpu-32.tar.gz
rm efficientdet_d0_coco17_tpu-32.tar.gz

git clone --depth 1 https://github.com/tensorflow/models.git
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
