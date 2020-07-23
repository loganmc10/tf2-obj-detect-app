#!/usr/bin/python3
import argparse
import os
import cv2
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from tensorflow.python.compiler.tensorrt import trt_convert as trt

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='path to image file')
parser.add_argument('-r', '--rt', action='store_true', help='enable TensorRT')
args = parser.parse_args()

model_name = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'

model_dir = model_name + '/saved_model'

if args.rt is True:
    if not os.path.exists('rt_model/' + model_name):
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_dir)
        converter.convert()
        converter.save('rt_model/' + model_name)
    model_dir = 'rt_model/' + model_name

image_path = args.file
image_np = cv2.imread(image_path)
(h, w) = image_np.shape[:2]
if w > h and h > 1080:
    r = 1080 / float(h)
    dim = (int(w * r), 1080)
    image_np = cv2.resize(image_np, dim, interpolation=cv2.INTER_AREA)
elif h > w and w > 1080:
    r = 1080 / float(w)
    dim = (1080, int(h * r))
    image_np = cv2.resize(image_np, dim, interpolation=cv2.INTER_AREA)

input_tensor = np.expand_dims(image_np, 0)
detect_fn = tf.saved_model.load(model_dir)
detections = detect_fn(input_tensor)

label_map_path = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)

viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np,
      detections['detection_boxes'][0].numpy(),
      detections['detection_classes'][0].numpy().astype(np.int32),
      detections['detection_scores'][0].numpy(),
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.40,
      agnostic_mode=False)

classes = detections['detection_classes'][0].numpy().astype(np.int32).tolist()
scores = detections['detection_scores'][0].numpy().tolist()
for i in range(len(scores)):
    if scores[i] > 0.40:
        print(category_index[classes[i]]['name'] + " " + str(scores[i]))

cv2.imwrite('pictures/output.png', image_np)
