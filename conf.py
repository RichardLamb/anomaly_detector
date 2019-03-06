import logging

##### general settings #####
PERFORM_CHECKS = True

INPUT_TYPE = 'video_stream'  # 'video_stream' / 'video_clip'
FILE_PATH = None
OUTPUT_PATH = None
MODEL = 'yolo'

##### overseer #####
HISTORIAN_MEMORY_DURATION = 300
HISTORIAN_NONE_MEMORY_MAXIMUM = 5# the number of consecutive iterations where None type observations are accepted before removing the tracker
HISTORIAN_PREDICTION_STRATEGY = 'sgolay_filter'# 'linear'/'fake_AR'/'sgolay_filter'
ANOMALY_DETECTION_THRESHOLD__N_STDS = 2
SGOLAY_POLYNOMIAL_ORDER = 3

if MODEL == 'yolo':
    TIME_SERIES_TRAIN_WINDOW = 6
else:
    TIME_SERIES_TRAIN_WINDOW = 12

if HISTORIAN_PREDICTION_STRATEGY == 'sgolay_filter':
    if TIME_SERIES_TRAIN_WINDOW % 2 != 0:
        TIME_SERIES_TRAIN_WINDOW -= 1
        logging.info('For the Savitsky-Golay filter, the window size setting must be an even number, changing to %s.' % (TIME_SERIES_TRAIN_WINDOW,))

observation_metadata_indices = {'q factor applicability': 0,
                                'value range minimum': 1,
                                'value range maximum': 2}

##### openpose | coco and body_25 #####
openpose_repo_path = 'openpose-master'
openpose_threshold = 0.1
openpose_inWidth = 368#pixel width in input layer of NN
