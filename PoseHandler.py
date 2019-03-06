import os
import numpy as np
import cv2

from conf import openpose_repo_path
from conf import openpose_threshold
from conf import openpose_inWidth


class Pose:
    def __init__(self, model):
        self.model = model
        self._get_model_files()
        self._get_pose_pairs()

        self.threshold = openpose_threshold#0.1
        self.inWidth = openpose_inWidth#368
        # self.inHeight = 224

        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)

    def _get_model_files(self):
        the_directory = os.path.join(openpose_repo_path, 'models', 'pose', '%s' % (self.model,))

        for file in os.listdir(the_directory):
            if file.endswith('.prototxt'):
                self.protoFile = os.path.join(the_directory, file)
            else:
                self.weightsFile = os.path.join(the_directory, file)

    def calculate(self, original_image, suppress=True):
        """
        Note: please don't delete the suppress=True.
        :param original_image:
        :param suppress:
        :return:
        """
        image = original_image.copy()
        frameHeight, frameWidth, channels = image.shape
        inHeight = self.inWidth#round(self.inWidth * (frameHeight/frameWidth))
        inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (self.inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inpBlob)
        output = self.net.forward()

        H = output.shape[2]
        W = output.shape[3]

        points = []
        for i in range(self.nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > self.threshold:
                points.append((int(x), int(y)))
            else:
                points.append((None, None))

        return self.draw_results(image, points), self.transform_points(points)

    def draw_results(self, original_image, points):
        image = original_image.copy()
        # image_skel = np.zeros((frameHeight, frameWidth, 3))
        for i, (x, y) in enumerate(points):
            if x is not None and y is not None:
                cv2.circle(image, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                    # cv2.circle(image_skel, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    # cv2.putText(image_skel, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        for pair in self.POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] != (None, None) and points[partB] != (None, None):
                cv2.line(image, points[partA], points[partB], (0, 255, 255), 3)
                # cv2.line(image_skel, points[partA], points[partB], (0, 255, 255), 3)
        return image

    @staticmethod
    def transform_points(points):
        transformed = []
        for el in points:
            if el is None:
                transformed.extend([None, None])
            else:
                transformed.extend([el[0], el[1]])
        return [[(el,) for el in transformed]]

    def _get_pose_pairs(self):
        if self.model == 'body_25':
            self.nPoints = 25
            self.POSE_PAIRS = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
                               [8, 12], [12, 13], [13, 14], [1, 0], [0, 15], [15, 17], [0, 16], [16, 18], [5, 18],
                               [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]]
        elif self.model == 'coco':
            self.nPoints = 18
            self.POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                               [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]
        elif self.model == 'mpi':
            self.nPoints = 15
            self.POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10],
                               [14, 11], [11, 12], [12, 13]]


class Stance:
    def __init__(self, joint_coords_list, body_model='body25'):
        self.joint_coords_list = joint_coords_list
        self.joint_ids = self._get_joint_ids(body_model)
        self.target_list = self._get_target_list(body_model)

    def _get_triad_ids(self, target_joint, ancillary_joint1, ancillary_joint2):
        return self.joint_ids[target_joint], self.joint_ids[ancillary_joint1], self.joint_ids[ancillary_joint2]

    def _get_pixel_coordinates(self, target_joints):
        return tuple(self.joint_coords_list[joint_id] for joint_id in self._get_triad_ids(*target_joints))

    @staticmethod
    def _get_joint_ids(body_model):
        if body_model == 'body25':
            return {
                'Nose': 0,
                'Neck': 1,
                'RShoulder': 2,
                'RElbow': 3,
                'RWrist': 4,
                'LShoulder': 5,
                'LElbow': 6,
                'LWrist': 7,
                'MidHip': 8,
                'RHip': 9,
                'RKnee': 10,
                'RAnkle': 11,
                'LHip': 12,
                'LKnee': 13,
                'LAnkle': 14,
                'REye': 15,
                'LEye': 16,
                'REar': 17,
                'LEar': 18,
                'LBigToe': 19,
                'LSmallToe': 20,
                'LHeel': 21,
                'RBigToe': 22,
                'RSmallToe': 23,
                'RHeel': 24,
                'Background': 25
            }

    @staticmethod
    def _get_target_list(body_model):
        """
        Add joint triad here by name in the form of a tuple.
        The first element must be the core joint of the triad (the angle is the angle between the element1-element2 and element1-element3 vectors)
        """
        if body_model == 'body25':
            return {1: ('LKnee', 'LHip', 'LAnkle'),
                    2: ('RKnee', 'RHip', 'RAnkle'),
                    3: ('MidHip', 'LHip', 'RHip'),
                    4: ('MidHip', 'LKnee', 'RKnee'),
                    5: ('LHeel', 'LKnee', 'LBigToe'),
                    6: ('RHeel', 'RKnee', 'RBigToe'),
                    7: ('Neck', 'RHip', 'RKnee'),
                    8: ('Neck', 'LHip', 'LKnee')
                    }

    def do_calculations(self):
        the_numbers = []
        for key, target in self.target_list.items():
            the_coordinates = self._get_pixel_coordinates(target)
            if any([el is None for el in the_coordinates]):
                # if the coordinate is missing
                the_numbers.append((None, (None, None)))

            else:
                the_joint = JointTriad(*the_coordinates)

                the_lengths = the_joint.get_lengths()
                if any([el == 0 for el in the_lengths]):
                    # to avoid division by zero when joints are overlapped
                    the_numbers.append((None, the_lengths))
                else:
                    the_angle = the_joint.get_angle()
                    the_numbers.append((the_angle, the_lengths))
        return the_numbers


class JointTriad:
    def __init__(self, target_joint, ancillary_joint1, ancillary_joint2):
        self.vectors = self._get_vectors(np.vstack((target_joint, ancillary_joint1, ancillary_joint2)))

    @staticmethod
    def _get_vectors(joint_coords_arr):
        return joint_coords_arr[1:3] - joint_coords_arr[0]

    def get_angle(self):
        if np.unique(self.vectors, axis=0).shape[0] != 2:
            # if the vectors are overlapped
            return None
        return np.round(np.rad2deg(np.arccos(
            np.dot(self.vectors[0], self.vectors[1]) /
            (np.linalg.norm(self.vectors[0]) * np.linalg.norm(self.vectors[1]))
        )), 1)

    def get_lengths(self):
        return np.linalg.norm(self.vectors[0]), np.linalg.norm(self.vectors[1])
