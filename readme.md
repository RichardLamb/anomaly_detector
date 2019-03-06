# Overseer Anomaly Correction

Detect and correct anomalies on the fly in time series data.
This was designed to accept the outputs of models like YOLO and OpenPose that typically perform their analysis on a frame-by frame basis.
The goal of this package is to improve the output of other models by considering prior observations.
In the context of multi-object detection where identification is not performed (such as YOLO), overseer can track objects by id. 

## Quick Start
### YOLO
1. Download the YOLO checkpoint (initial-model.h5) from: https://github.com/ndaidong/yolo-person-detect
2. Put the file in the yolo folder.

### OpenPose
1. Clone the OpenPose repository: https://github.com/CMU-Perceptual-Computing-Lab/openpose
2. Download the OpenPose coco and/or body25 weights by running getModels (located in the models folder of the openpose repository).
3. Add the path to openpose-master to the conf.py file (the openpose_repo_path variable).

### To run
Run front_end.py

By default it streams images from a webcam to YOLO.
