#!/usr/bin/env python3

import glob
import argparse
import pickle

import cv2

from os import path, mkdir
from shutil import rmtree
from random import choice

from tqdm import tqdm

from utils.network import yolo
from utils.detector import detect
from utils.drawer import draw_boxes

min_threshold = 0.5

model_file = 'initial-model.h5'
image_dir = 'tests/images'

video_exts = ['.avi', '.mp4', '.mkv']

model = yolo()
model.load_weights(model_file)


def _predict(image, suppress=True):

    image = cv2.resize(image, (416, 416))

    boxes, labels = detect(image, model)

    image, box_data = draw_boxes(image, boxes, labels)

    image = cv2.resize(image, (800, 600))

    if len(box_data) > 0:

        for i in range(len(box_data)):

            if not suppress:
                # print('x_min:', box_data[i][0][0],
                #       'x_max:', box_data[i][1][0],
                #       'y_min:', box_data[i][2][0],
                #       'y_max:', box_data[i][3][0],
                #       'x_c:', box_data[i][4][0],
                #       'y_c:', box_data[i][5][0],
                #       'prob', box_data[i][6][0])

                box_image = image[box_data[i][2][0]:box_data[i][3][0], box_data[i][0][0]:box_data[i][1][0]]

                #Added condition as sometimes crashed
                if box_image.shape[0] > 0 and box_image.shape[1] > 0:
                    cv2.imshow(str(i), box_image)
                    cv2.imshow('image', image)

    return image, box_data


def predict_image(image_path):
    image = cv2.imread(image_path)
    image = _predict(image)[0]

    while True:
        k = cv2.waitKey(30)
        if k == 27:
            break
        cv2.imshow('Image prediction', image)
    cv2.destroyAllWindows()


def predict_multi(images, output):
    print('Founded {} images. Start handling...'.format(len(images)))
    for img_path in tqdm(images):
        image = cv2.imread(img_path)
        image = _predict(image)[0]
        fname = path.basename(img_path)
        f = output + '/' + fname
        print('Finish handling "{}"'.format(fname))
        cv2.imwrite(f, image)


def predict_video(video_path, pickle_name="box_data.pickle",  suppress=True):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0

    if not cap.isOpened():
        print('Fail to load video "{}" file'.format(video_path))
    video_box_data = list()

    while cap.isOpened():
        i += 1

        # if i % 10 == 0:
        #     print('\r%s out of %s' % (i, frame_count), end='', flush=True)

        ret, frame = cap.read()

        if ret is False:
            break
        k = cv2.waitKey(30)
        if k == 27:
            break
        # RL code
        frame, box_data = _predict(frame)

        video_box_data.append(box_data)
        pickle_out = open(pickle_name, "wb")
        pickle.dump(video_box_data, pickle_out)
        pickle_out.close()

        if not suppress:
            cv2.imshow('Video prediction', frame)


    cap.release()
    cv2.destroyAllWindows()


def check(f=None, o=None):
    if isinstance(f, int):
        return predict_video(f)

    if not f:
        images = glob.glob(image_dir + '/*.jpg')
        f = choice(images)

    if not path.exists(f):
        return print('File/folder not found: "{}"'.format(f))

    if path.isfile(f):
        ext = path.splitext(f)[1]
        if ext in video_exts:
            return predict_video(f)
        return predict_image(f)
    if path.isdir(f):
        if path.exists(o):
            rmtree(o)
        mkdir(o)
        images = glob.glob(f + '/*.jpg')
        return predict_multi(images, o)


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--file',
        help='Image file to predict'
    )
    parser.add_argument(
        '-c',
        '--cam',
        help='Camera source to predict'
    )
    parser.add_argument(
        '-d',
        '--dir',
        help='Image dir to predict'
    )
    parser.add_argument(
        '-o',
        '--output',
        help='Image dir to export the output'
    )

    args = parser.parse_args()

    if args.cam:
        check(int(args.cam))
    elif args.dir:
        check(path.normpath(args.dir), path.normpath(args.output))
    elif args.file:
        check(path.normpath(args.file))
    else:
        check()


if __name__ == '__main__':
    start()
