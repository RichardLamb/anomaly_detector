import cv2
import numpy as np

paint_color = (0, 0, 255)
paint_size = 1

text_color = (0, 0, 255)
text_font = cv2.FONT_HERSHEY_PLAIN

font_size = 1.0
font_weight = 1

# Define colours

colour_list = [
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255),
    (255, 0, 0),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128)
]

def draw_boxes(image, boxes, labels, suppress=False):

    box_data = list()

    for i, box in enumerate(boxes):


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
        x_c = int(box.x * image.shape[1])
        y_c = int(box.y * image.shape[0])

        box_data.append([(xmin,), (xmax,), (ymin,), (ymax,), (x_c,), (y_c,), (score,)])

        if not suppress:
            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                colour_list[i % len(colour_list)],
                paint_size
            )
            cv2.putText(
                image,
                'id: ' + str(i),
                (xmin, ymin - 12),
                text_font,
                font_size,
                colour_list[i % len(colour_list)],
                font_weight,
                cv2.LINE_AA
            )

    return image, box_data
