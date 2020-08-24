#!/usr/bin/python3
import io
import boto3
import json
import secrets
import collections
import time
import argparse
import os
import cv2
import hashlib
import urllib.request
import tensorflow as tf
import numpy as np
import paho.mqtt.client as mqtt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from azure.storage.blob import ContainerClient, ContentSettings

def on_connect(client, userdata, flags, rc):
    print("MQTT connection returned result: "+ mqtt.connack_string(rc))

mqttc = mqtt.Client()
mqttc.on_connect = on_connect
mqttc.tls_set()
mqttc.username_pw_set(os.getenv('MQTT_USER'), password=os.getenv('MQTT_PASSWORD'))
mqttc.connect(os.getenv('MQTT_BROKER'), port=8883)
mqttc.loop_start()

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--blob', help='blob type (s3 or azure). Defaults to s3', default="s3", type=str)
parser.add_argument('-i', '--input', help='path to video feed. Format: "http://feedone,feed1_name http://feedtwo,feed2_name"', type=str)
parser.add_argument('-r', '--rt', action='store_true', help='enable TensorRT')
parser.add_argument('-s', '--imageset', help='Imageset to use (coco or oid). Defaults to coco', default="coco", type=str)
parser.add_argument('-f', '--freq', help='Analysis frequency in seconds. Defaults to 10', default=10, type=int)
parser.add_argument('-t', '--threshold', help='detection threshold. Defaults to 0.40', default=0.40, type=float)
args = parser.parse_args()

if args.blob == "s3":
    s3 = boto3.client('s3')
elif args.blob == "azure":
    az_container = ContainerClient(os.getenv('AZURE_CONTAINER_URL'), os.getenv('AZURE_CONTAINER_NAME'), credential=os.getenv('AZURE_CREDENTIAL'))

if args.imageset == "coco":
    model_name = 'efficientdet_d7_coco17_tpu-32'
    label_map_path = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
elif args.imageset == "oid":
    model_name = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12'
    label_map_path = 'models/research/object_detection/data/oid_v4_label_map.pbtxt'
model_dir = model_name + '/saved_model'

if args.rt is True:
    if not os.path.exists('rt_model/' + model_name):
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_dir)
        converter.convert()
        converter.save('rt_model/' + model_name)
    model_dir = 'rt_model/' + model_name

category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

detect_fn = tf.saved_model.load(model_dir)
if args.imageset == "oid": # Needed because it is a TF1 Model
    detect_fn =  detect_fn.signatures['serving_default']

num_feeds = len(args.input.split())
cap = []
type = []
names = []
for i in range(num_feeds):
    if ".jpg" not in args.input.split()[i].split(',')[0]:
        cap.append(cv2.VideoCapture(args.input.split()[i].split(',')[0]))
        type.append("video")
    else:
        type.append("image")
    names.append(args.input.split()[i].split(',')[1])
last_time = 0
last_hash = ""

try:
    while True:
        sleep_time = time.time() - last_time
        if sleep_time < args.freq:
            time.sleep(args.freq - sleep_time)
        last_time = time.time()

        for j in range(num_feeds):
            if type[j] == "video":
                cap[j].open(args.input.split()[j].split(',')[0])
                ret, image_np = cap[j].read()
                if ret is False:
                    continue
            else:
                try:
                    image_file = urllib.request.urlopen(args.input.split()[j].split(',')[0], 10)
                except urllib.error.HTTPError:
                    continue
                image_bytes = image_file.read()
                current_hash = hashlib.md5(image_bytes).hexdigest()
                if current_hash == last_hash:
                    continue
                else:
                    last_hash = current_hash
                image = np.asarray(bytearray(image_bytes), dtype="uint8")
                image_np = cv2.imdecode(image, cv2.IMREAD_COLOR)

            if args.imageset == "coco":
                input_tensor = np.expand_dims(image_np, 0)
            elif args.imageset == "oid":
                input_tensor = tf.convert_to_tensor(image_np)
                input_tensor = input_tensor[tf.newaxis, ...]

            detections = detect_fn(input_tensor)

            classes = detections['detection_classes'][0].numpy().astype(np.int32).tolist()
            scores = detections['detection_scores'][0].numpy().tolist()
            items = []
            for i in range(len(scores)):
                if scores[i] > args.threshold:
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
              min_score_thresh=args.threshold,
              line_thickness=2)

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
            file_name = secrets.token_hex(32) + ".jpg"
            if args.blob == "s3":
                s3.upload_fileobj(io_buf, os.getenv('S3_BUCKET_NAME'), file_name, ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/jpeg'})
            elif args.blob == "azure":
                az_container.upload_blob(file_name, io_buf, content_settings=ContentSettings(content_type='image/jpeg'))

            occurrences = collections.Counter(items)
            occurrences['file_name'] = '"' + file_name + '"'
            mqttc.publish(names[j], payload=json.dumps(occurrences))

except KeyboardInterrupt:
    for i in range(num_feeds):
        if type[i] == "video":
            cap[i].release()
    mqttc.loop_stop()
    mqttc.disconnect()
