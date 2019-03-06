import os
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import xml.etree.ElementTree as ET

from .BoundBox import BoundBox, bbox_iou


def normalize(image):
    image = image / 255.

    return image


def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = set()

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                all_imgs += [img]
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        seen_labels.add(obj['name'])

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

    return all_imgs, seen_labels


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def reset(self):
        self.offset = 4


class BatchGenerator:
    def __init__(
        self,
        images,
        config,
        shuffle=True,
        jitter=True,
        norm=True
    ):

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        self.anchors = [
            BoundBox(
                0,
                0,
                config['ANCHORS'][2*i],
                config['ANCHORS'][2*i+1]
            ) for i in range(int(len(config['ANCHORS'])/2))
        ]

        def sometimes(aug):
            return iaa.Sometimes(0.5, aug)

        self.aug_pipe = iaa.Sequential(
            [
                sometimes(iaa.Affine()),
                iaa.SomeOf(
                    (0, 5),
                    [
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)),
                            iaa.AverageBlur(k=(2, 7)),
                            iaa.MedianBlur(k=(3, 11)),
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                        iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.05*255),
                            per_channel=0.5
                        ),
                        iaa.OneOf([
                            iaa.Dropout(
                                (0.01, 0.1),
                                per_channel=0.5
                            ),
                        ]),
                        iaa.Add((-10, 10), per_channel=0.5),
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

        if shuffle:
            np.random.shuffle(self.images)

    def get_generator(self):
        num_img = len(self.images)

        total_count = 0
        batch_count = 0

        batch_size = self.config['BATCH_SIZE']

        x_batch = np.zeros((
            batch_size,
            self.config['IMAGE_H'],
            self.config['IMAGE_W'],
            3
        ))
        b_batch = np.zeros((
            batch_size,
            1,
            1,
            1,
            self.config['TRUE_BOX_BUFFER'],
            4
        ))
        y_batch = np.zeros((
            batch_size,
            self.config['GRID_H'],
            self.config['GRID_W'],
            self.config['BOX'],
            4 + 1 + 1
        ))

        while True:
            if total_count < num_img:
                train_instance = self.images[total_count]
                img, all_objs = self.aug_image(
                    train_instance,
                    jitter=self.jitter
                )
                true_box_index = 0

                for obj in all_objs:
                    xmax = obj['xmax']
                    xmin = obj['xmin']
                    ymax = obj['ymax']
                    ymin = obj['ymin']
                    if xmax > xmin and ymax > ymin \
                            and obj['name'] in self.config['LABELS']:
                        center_x = .5*(xmin + xmax)
                        center_x = center_x / (
                            float(
                                self.config['IMAGE_W']
                            ) / self.config['GRID_W']
                        )
                        center_y = .5*(ymin + ymax)
                        center_y = center_y / (
                            float(
                                self.config['IMAGE_H']
                            ) / self.config['GRID_H']
                        )

                        grid_x = int(np.floor(center_x))
                        grid_y = int(np.floor(center_y))

                        if grid_x < self.config['GRID_W'] \
                                and grid_y < self.config['GRID_H']:
                            obj_indx = self.config['LABELS'].index(obj['name'])

                            center_w = (xmax - xmin) / (
                                float(
                                    self.config['IMAGE_W']
                                ) / self.config['GRID_W']
                            )
                            center_h = (ymax - ymin) / (
                                float(
                                    self.config['IMAGE_W']
                                ) / self.config['GRID_W']
                            )

                            box = [center_x, center_y, center_w, center_h]

                            best_anchor = -1
                            max_iou = -1

                            shifted_box = BoundBox(0,
                                                   0,
                                                   center_w,
                                                   center_h)

                            for i in range(len(self.anchors)):
                                anchor = self.anchors[i]
                                iou = bbox_iou(shifted_box, anchor)

                                if max_iou < iou:
                                    best_anchor = i
                                    max_iou = iou

                            y_batch[
                                batch_count,
                                grid_y,
                                grid_x,
                                best_anchor,
                                0:4
                            ] = box
                            y_batch[
                                batch_count,
                                grid_y,
                                grid_x,
                                best_anchor,
                                4
                            ] = 1.
                            y_batch[
                                batch_count,
                                grid_y,
                                grid_x,
                                best_anchor,
                                5
                            ] = obj_indx

                            # assign the true box to b_batch
                            b_batch[batch_count, 0, 0, 0, true_box_index] = box

                            true_box_index += 1
                            true_box_buffer = self.config['TRUE_BOX_BUFFER']
                            true_box_index = true_box_index % true_box_buffer
                # assign input image to x_batch
                if self.norm:
                    x_batch[batch_count] = normalize(img)
                else:
                    # plot image and bounding boxes for sanity check
                    for obj in all_objs:
                        xmax = obj['xmax']
                        xmin = obj['xmin']
                        ymax = obj['ymax']
                        ymin = obj['ymin']
                        if xmax > xmin and ymax > ymin:
                            cv2.rectangle(
                                img[:, :, ::-1],
                                (xmin, ymin),
                                (xmax, ymax),
                                (255, 0, 0),
                                3
                            )
                            cv2.putText(
                                img[:, :, ::-1],
                                obj['name'],
                                (xmin + 2, ymin + 12),
                                0,
                                1.2e-3 * img.shape[0],
                                (0, 255, 0),
                                2
                            )

                    x_batch[batch_count] = img

                # increase instance counter in current batch
                batch_count += 1

            total_count += 1
            if total_count >= num_img:
                total_count = 0
                if self.shuffle:
                    np.random.shuffle(self.images)

            if batch_count >= batch_size:
                yield [x_batch, b_batch], y_batch

                x_batch = np.zeros((
                    batch_size,
                    self.config['IMAGE_H'],
                    self.config['IMAGE_W'],
                    3
                ))
                y_batch = np.zeros((
                    batch_size,
                    self.config['GRID_H'],
                    self.config['GRID_W'],
                    self.config['BOX'],
                    5 + self.config['CLASS']
                ))

                batch_count = 0

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']

        image = cv2.imread(image_name)
        h, w, c = image.shape

        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(
                image,
                (0, 0),
                fx=scale,
                fy=scale
            )

            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy: (offy + h), offx: (offx + w)]

            flip = np.random.binomial(1, .5)
            if flip > 0.5:
                image = cv2.flip(image, 1)

            image = self.aug_pipe.augment_image(image)

        # resize the image to standard size
        image = cv2.resize(
            image,
            (self.config['IMAGE_H'], self.config['IMAGE_W'])
        )
        image = image[:, :, ::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter:
                    obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)

            for attr in ['ymin', 'ymax']:
                if jitter:
                    obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin

        return image, all_objs

    def get_dateset_size(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))
