import numpy as np


from .BoundBox import BoundBox, bbox_iou


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def decode_netout(netout, obj_threshold, nms_threshold, anchors, nb_class):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    netout[..., 4] = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] \
        * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:
                    x, y, w, h = netout[row, col, b, :4]

                    x = (col + sigmoid(x)) / grid_w
                    y = (row + sigmoid(y)) / grid_h
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h
                    confidence = netout[row, col, b, 4]

                    box = BoundBox(x, y, w, h, confidence, classes)

                    boxes.append(box)

    for c in range(nb_class):
        sorted_indices = list(
            reversed(
                np.argsort([box.classes[c] for box in boxes])
            )
        )

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    box_i = boxes[index_i]
                    box_j = boxes[index_j]
                    if bbox_iou(box_i, box_j) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes
