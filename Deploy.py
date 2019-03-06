import os
import sys
import logging


def start_logs():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create a file handler
    file_handler = logging.FileHandler('logs.log')
    file_handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))# For printing to stdout


def check_model_requirements(model):
    fail = False
    if model == 'yolo':
        if not os.path.isfile(os.path.join(os.getcwd(), 'yolo', 'yolo.weights')):
            logging.warning('YOLO weights missing. Please download yolo.weights into the yolo directory.')
            fail = True
    elif model == 'coco':
        if not os.path.isfile(os.path.join(os.getcwd(), 'openpose-master', 'models', 'pose', 'coco', 'pose_iter_440000.caffemodel')):
            logging.warning('OpenPose coco weights missing. Please clone openpose and run the getModels batch or shell file in the model directory to download the weights.')
            fail = True
    elif model == 'body_25':
        if not os.path.isfile(os.path.join(os.getcwd(), 'openpose-master', 'models', 'pose', 'body_25', 'pose_iter_584000.caffemodel')):
            logging.warning('OpenPose body 25 weights missing. Please clone openpose and run the getModels batch or shell file in the model directory to download the weights.')
            fail = True
    if fail:
        sys.exit('Missing Files. Check logs for more info.')
