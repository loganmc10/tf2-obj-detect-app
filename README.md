# tf2-obj-detect-app

```
usage: tf_app.py [-h] [-b BLOB] [-i INPUT] [-r] [-s IMAGESET] [-f FREQ] [-t THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -b BLOB, --blob BLOB  blob type (s3 or azure). Defaults to s3
  -i INPUT, --input INPUT
                        path to video feed. Format: "http://feedone,feed1_name http://feedtwo,feed2_name"
  -r, --rt              enable TensorRT
  -s IMAGESET, --imageset IMAGESET
                        Imageset to use (coco or oid). Defaults to coco
  -f FREQ, --freq FREQ  Analysis frequency in seconds. Defaults to 10
  -t THRESHOLD, --threshold THRESHOLD
                        detection threshold. Defaults to 0.40
```

## Workflow
1. OpenCV captures a video feed (a webcam for example)
2. Image is passed to Tensorflow for object detection analysis (by default this happens every 10 seconds)
3. Tensorflow determines what objects are present in the image
4. If objects are present, the image is uploaded to Amazon S3 (images retained for 35 days)
5. Object labels and image URL are sent via MQTT to a broker, message is collected by Node-RED, which forwards the data to InfluxDB
6. Grafana graphs the labels, attaching the image URL to each data point as metadata

## Final result
* Object labels graphed in time series

![115850073_598299394457821_2554421986432986060_o](https://user-images.githubusercontent.com/848146/88460651-eefa9700-ce5a-11ea-91fb-44f07e8439a8.png)
* Image URL metadata attached to each data point

![115865554_598299621124465_7487020321069107523_n](https://user-images.githubusercontent.com/848146/88460650-eefa9700-ce5a-11ea-9daa-e71417d0b854.png)
* Image with Tensorflow object detection overlay can be display via the graph link

![115960383_598299894457771_2460616232917877997_o](https://user-images.githubusercontent.com/848146/88460648-ebffa680-ce5a-11ea-8051-d513211369ff.jpg)
