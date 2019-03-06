import cv2

from yolo.utils.network import yolo
from yolo.utils.detector import detect
from yolo.utils.drawer import draw_boxes

# Ensure initial-model.h5 and yolo.weights file are added to the yolo folder

paint_color = (0, 0, 255)
paint_size = 1

text_color = (0, 0, 255)
text_font = cv2.FONT_HERSHEY_PLAIN

font_size = 1.0
font_weight = 1


class Yolo:
    def __init__(self):
        self.model = yolo()
        self.min_threshold = 0.5
        self.model_file = 'yolo/initial-model.h5'
        self.model.load_weights(self.model_file)

        # self.inWidth = 416

    def _get_boxes(self, image, suppress=True):
        """
        image: numpy array of image
        suppress: if set to False, will display:
                            1) Frame with all bounding boxes
                            2) Each box with contents in separate window

        returns:
        box_data: co-ordinates of each bounding box in image (list of lists)
        """

        image = cv2.resize(image, (416, 416))

        boxes, labels = detect(image, self.model)

        image, box_data = draw_boxes(image, boxes, labels)

        if len(box_data) > 0:

            for i in range(len(box_data)):

                if not suppress:
                    print('x_min:', box_data[i][0][0],
                          'x_max:', box_data[i][1][0],
                          'y_min:', box_data[i][2][0],
                          'y_max:', box_data[i][3][0],
                          'x_c:', box_data[i][4][0],
                          'y_c:', box_data[i][5][0],
                          'prob', box_data[i][6][0])

                    box_image = image[box_data[i][2][0]:box_data[i][3][0], box_data[i][0][0]:box_data[i][1][0]]

                    # Added if condition as sometimes crashed with size = 0
                    if box_image.shape[0] > 0 and box_image.shape[1] > 0:

                        while True:
                            k = cv2.waitKey(30)
                            if k == 27:
                                break
                            cv2.imshow('Image prediction', image)
                            cv2.imshow(str(i), box_image)

                        cv2.destroyAllWindows()

        return image, box_data

    def calculate(self, image, suppress=True):
        return self._get_boxes(image=image, suppress=suppress)

    def draw_results(self, image, boxes, suppress=True):
        """
        image: numpy array of image
        boxes: co-ordinates of each bounding box in image (list of lists)
        suppress: if set to False, will display:
                            1) Frame with all bounding boxes

        returns:
        final_image: image with bounding boxes added
        """

        frameHeight, frameWidth, channels = image.shape
        image = cv2.resize(image, (416, 416))

        # Define colours

        colour_list = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 255, 255),
            (0, 0, 255),
            (255, 0, 255),
            (255, 255, 0),
            (128, 128, 0),
            (128, 0, 128),
            (0, 128, 128)
        ]

        # Define keys
        box_dict = {
            'x_min': 0,
            'x_max': 1,
            'y_min': 2,
            'y_max': 3,
            'x_centre': 4,
            'y_centre': 5,
            'score': 6,
            'q_factor': 8
        }

        if len(boxes.keys()) != 0:
            for k, box in boxes.items():


                x_min = int(box[box_dict['x_min']][0])
                x_max = int(box[box_dict['x_max']][0])
                y_min = int(box[box_dict['y_min']][0])
                y_max = int(box[box_dict['y_max']][0])
                score = box[box_dict['score']][0]
                q_factor = box[box_dict['q_factor']][0]

                cv2.rectangle(
                    image,
                    (x_min, y_min),
                    (x_max, y_max),
                    colour_list[k % len(colour_list)],
                    paint_size
                )

                cv2.putText(
                    image,
                    'id: %s | L2: %.1E' % (k, q_factor),
                    (x_min, y_min - 12),
                    text_font,
                    font_size,
                    colour_list[k % len(colour_list)],
                    font_weight,
                    cv2.LINE_AA
                )

        final_image = cv2.resize(image, (frameWidth, frameHeight))

        if not suppress:
            while True:
                k = cv2.waitKey(30)
                if k == 27:
                    break
                cv2.imshow('Image prediction', final_image)

            cv2.destroyAllWindows()

        return final_image
