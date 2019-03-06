import numpy as np
from keras import models
from keras import layers
import tensorflow as tf

LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']  # nopep8

IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W = 13, 13
BOX = 5
CLASS = len(LABELS)
CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3
ANCHORS = [
    0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
    5.47434, 7.88282, 3.52778, 9.77052, 9.16828
]

NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0

BATCH_SIZE = 16
WARM_UP_BATCHES = 0
TRUE_BOX_BUFFER = 50

ALPHA = 0.1

input_image = layers.Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes = layers.Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))


def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)


def yolo():

    # Layer 1
    x = layers.Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_1',
        use_bias=False
    )(input_image)

    x = layers.BatchNormalization(name='norm_1')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    # Layer 2
    x = layers.Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_2',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_2')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = layers.Conv2D(
        128,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_3',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_3')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 4
    x = layers.Conv2D(
        64,
        (1, 1),
        strides=(1, 1),
        padding='same',
        name='conv_4',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_4')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 5
    x = layers.Conv2D(
        128,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_5',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_5')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = layers.Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_6',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_6')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 7
    x = layers.Conv2D(
        128,
        (1, 1),
        strides=(1, 1),
        padding='same',
        name='conv_7',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_7')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 8
    x = layers.Conv2D(
        256,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_8',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_8')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = layers.Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_9',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_9')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 10
    x = layers.Conv2D(
        256,
        (1, 1),
        strides=(1, 1),
        padding='same',
        name='conv_10',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_10')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 11
    x = layers.Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_11',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_11')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 12
    x = layers.Conv2D(
        256,
        (1, 1),
        strides=(1, 1),
        padding='same',
        name='conv_12',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_12')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 13
    x = layers.Conv2D(
        512,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_13',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_13')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    skip_connection = x

    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    # Layer 14
    x = layers.Conv2D(
        1024,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_14',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_14')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 15
    x = layers.Conv2D(
        512,
        (1, 1),
        strides=(1, 1),
        padding='same',
        name='conv_15',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_15')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 16
    x = layers.Conv2D(
        1024,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_16',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_16')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 17
    x = layers.Conv2D(
        512,
        (1, 1),
        strides=(1, 1),
        padding='same',
        name='conv_17',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_17')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 18
    x = layers.Conv2D(
        1024,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_18',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_18')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 19
    x = layers.Conv2D(
        1024,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_19',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_19')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 20
    x = layers.Conv2D(
        1024,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_20',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_20')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 21
    skip_connection = layers.Conv2D(
        64,
        (1, 1),
        strides=(1, 1),
        padding='same',
        name='conv_21',
        use_bias=False
    )(skip_connection)

    skip_connection = layers.BatchNormalization(
        name='norm_21'
    )(skip_connection)

    skip_connection = layers.advanced_activations.LeakyReLU(
        alpha=ALPHA
    )(skip_connection)

    skip_connection = layers.Lambda(
        space_to_depth_x2
    )(skip_connection)

    x = layers.concatenate([skip_connection, x])

    # Layer 22
    x = layers.Conv2D(
        1024,
        (3, 3),
        strides=(1, 1),
        padding='same',
        name='conv_22',
        use_bias=False
    )(x)
    x = layers.BatchNormalization(name='norm_22')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Layer 23
    x = layers.Conv2D(
        (4 + 1 + CLASS) * 5,
        (1, 1),
        strides=(1, 1),
        padding='same',
        name='conv_23'
    )(x)
    output = layers.Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

    output = layers.Lambda(lambda args: args[0])([output, true_boxes])

    model = models.Model([input_image, true_boxes], output)

    return model
