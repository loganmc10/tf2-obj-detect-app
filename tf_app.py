#!/usr/bin/python3
import io
import boto3
import json
import collections
import time
import argparse
import os
import cv2
import tensorflow as tf
import numpy as np
import paho.mqtt.client as mqtt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from tensorflow.python.compiler.tensorrt import trt_convert as trt

THRESHOLD = 0.40

def on_connect(client, userdata, flags, rc):
    print("MQTT connection returned result: "+ mqtt.connack_string(rc))

s3 = boto3.client('s3')
mqttc = mqtt.Client()
mqttc.on_connect = on_connect
mqttc.tls_set()
mqttc.username_pw_set("mqtt", password=os.getenv('MQTT_PASSWORD'))
mqttc.connect("iot.bacoosta.com", port=8883)
mqttc.loop_start()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to video feed')
parser.add_argument('-r', '--rt', action='store_true', help='enable TensorRT')
parser.add_argument('-t', '--type', help='Imageset type (coco or oid). Defaults to coco', default="coco")
parser.add_argument('-f', '--freq', help='Analysis frequency in seconds. Defaults to 10', default=10)
args = parser.parse_args()

if args.type == "coco":
    model_name = 'efficientdet_d7_coco17_tpu-32'
    label_map_path = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
elif args.type == "oid":
    model_name = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12'
    label_map_path = 'models/research/object_detection/data/oid_v4_label_map.pbtxt'
model_dir = model_name + '/saved_model'

if args.rt is True:
    if not os.path.exists('rt_model/' + model_name):
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_dir)
        converter.convert()
        converter.save('rt_model/' + model_name)
    model_dir = 'rt_model/' + model_name

label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detect_fn = tf.saved_model.load(model_dir)
if args.type == "oid": # Needed because it is a TF1 Model
    detect_fn =  detect_fn.signatures['serving_default']

cap = cv2.VideoCapture(args.input)
last_time = 0

try:
    while True:
        sleep_time = time.time() - last_time
        if sleep_time < args.freq:
            time.sleep(args.freq - sleep_time)
        last_time = time.time()

        ret, image_np = cap.read()

        if args.type == "coco":
            input_tensor = np.expand_dims(image_np, 0)
        elif args.type == "oid":
            input_tensor = tf.convert_to_tensor(image_np)
            input_tensor = input_tensor[tf.newaxis, ...]

        detections = detect_fn(input_tensor)

        classes = detections['detection_classes'][0].numpy().astype(np.int32).tolist()
        scores = detections['detection_scores'][0].numpy().tolist()
        items = []
        for i in range(len(scores)):
            if scores[i] > THRESHOLD:
                items.append(category_index[classes[i]]['name'].replace(' ', '_'))

        if len(items) == 0:
            continue

        viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np,
              detections['detection_boxes'][0].numpy(),
              detections['detection_classes'][0].numpy().astype(np.int32),
              detections['detection_scores'][0].numpy(),
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=THRESHOLD,
              agnostic_mode=False)

        (h, w) = image_np.shape[:2]
        if w > h and h > 1080:
            r = 1080 / float(h)
            dim = (int(w * r), 1080)
            image_np = cv2.resize(image_np, dim, interpolation=cv2.INTER_AREA)
        elif h > w and w > 1080:
            r = 1080 / float(w)
            dim = (1080, int(h * r))
            image_np = cv2.resize(image_np, dim, interpolation=cv2.INTER_AREA)

        result, image = cv2.imencode('.JPEG', image_np)
        io_buf = io.BytesIO(image)
        file_name = str(time.time_ns()) + ".jpg"
        s3.upload_fileobj(io_buf, "iotcameraapp", file_name, ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/jpeg'})

        occurrences = collections.Counter(items)
        occurrences['file_name'] = '"' + file_name + '"'
        mqttc.publish("camera", payload=json.dumps(occurrences))

except KeyboardInterrupt:
    cap.release()
    mqttc.loop_stop()
    mqttc.disconnect()
