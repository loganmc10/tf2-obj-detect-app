#!/usr/bin/python3
import sys
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from six import BytesIO
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

model_name = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'

model_dir = model_name + '/saved_model'

if len(sys.argv) > 2:
    if sys.argv[2] == "RT" and not os.path.exists('rt_model/' + model_name):
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_dir)
        converter.convert()
        model_dir = 'rt_model/' + model_name
        converter.save(model_dir)

def load_image_into_numpy_array(path):
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

image_path = sys.argv[1]
image_np = load_image_into_numpy_array(image_path)
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

plt.rcParams['figure.figsize'] = [42, 21]
image_np_with_detections = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'][0].numpy(),
      detections['detection_classes'][0].numpy().astype(np.int32),
      detections['detection_scores'][0].numpy(),
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.40,
      agnostic_mode=False)
plt.imshow(image_np_with_detections)
plt.show()
