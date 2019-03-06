import cv2
import numpy as np

paint_color = (0, 0, 255)
paint_size = 1

text_color = (0, 0, 255)
text_font = cv2.FONT_HERSHEY_PLAIN

font_size = 1.0
font_weight = 1


def draw_boxes(image, boxes, labels):

    for box in boxes:
        label_id = box.get_label()
        if label_id != 0:
            continue

        label = labels[label_id]

        score = box.get_score()
        score = np.around(score, 2)

        xmin = int((box.x - box.w/2) * image.shape[1])
        xmax = int((box.x + box.w/2) * image.shape[1])
        ymin = int((box.y - box.h/2) * image.shape[0])
        ymax = int((box.y + box.h/2) * image.shape[0])

        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            paint_color,
            paint_size
        )
        cv2.putText(
            image,
            label + ' ' + str(score),
            (xmin, ymin - 12),
            text_font,
            font_size,
            text_color,
            font_weight,
            cv2.LINE_AA
        )

    return image
