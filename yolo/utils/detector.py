import numpy as np
import cv2

from .BoundBox import BoundBox, bbox_iou

from .network import TRUE_BOX_BUFFER, OBJ_THRESHOLD, \
    NMS_THRESHOLD, ANCHORS, CLASS, LABELS


from .helpers import softmax, sigmoid, decode_netout


def detect(input_image, model, obj_threshold=OBJ_THRESHOLD):

    input_image = input_image / 255.
    input_image = input_image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)

    dummy_array = np.zeros((1, 1, 1, 1, TRUE_BOX_BUFFER, 4))
    netout = model.predict([input_image, dummy_array])

    boxes = decode_netout(
        netout[0],
        obj_threshold=obj_threshold,
        nms_threshold=NMS_THRESHOLD,
        anchors=ANCHORS,
        nb_class=CLASS
    )

    return boxes, LABELS
