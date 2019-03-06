import logging
import os
import numpy as np
import cv2

from VideoHandler import VideoHandler

from Overseer import Overseer

from PoseHandler import Pose
from YoloHandler import Yolo


class SimulatedUser:
    def __init__(self, data_source, model, filepath=None, output_path=None):
        self.video_handler = VideoHandler(data_source=data_source, filepath=filepath, output_path=output_path)
        self.input_data = self.video_handler.data_generator()

        self.output_path = output_path

        self.get_model(model=model)
        self.do_exploratory_work()

        self.get_help()

    def get_model(self, model):
        """

        :param model:
        :return:
        """
        self.model_name = model
        if model == 'yolo':
            self.model = Yolo()
        elif model in ['coco', 'body_25']:
            self.model = Pose(model)
        else:
            logging.warning('ERROR: No valid model selected.')

    def get_help(self):
        """
        Creates the observation correction object.
        :return:
        """
        self.the_help = Overseer()

    def use_lots(self):
        frame_count = 0
        while True:
            self.use(frame_count)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if self.video_handler.data_source == 'video_stream':
                    self.get_image()
                break

        self.video_handler.kill_stream()
        cv2.destroyAllWindows()

    def use(self, frame_count):
        # get some data
        input_image = self.get_image()

        # get the output of the model of choice
        output_image_default, observations = self.get_observations(input_image=input_image)
        # at this point the preferred format is a list of lists
            # the first list level contains a sublist for each different complete object in the data
                #eg: one sublist for each person in an image
            # the second list level contains the actual observations for a given object
                #eg: the observations for a given person in an image

        # make new observations or engineer features
        observations = self.engineer_features(observations)

        # convert to the correct format and write metadata
        observations = self.write_metadata(observations)
        # metadata can assist the correction and is constructed as follows:
            # an observation is a tuple of the following format: (observation, (q_flag, value1, value2))
                # element 1 of the subtuple is a boolean which asserts inclusion into the q factor calculation
                # element 2 of the subtuple is a float that defines the minimum possible value of the feature
                # element 3 of the subtuple is a float that defines the maximum possible value of the feature
                # if no value is desired or available, use None

        # an example of the final input type is:
            # [[(0,(True,None,None)),(1,(False,None,None))],[(100,(True,None,None)),(200,(False,None,None))]]
            # this describes two objects that each have two observations
                # For the first object, the 0 and 1 values are the actual observations provided by the input model
                # For the second object, the 100 and 200 values are the actual observations provided by the input model

        # this calls the observation correction module
        corrected_observations = self.the_help.get_corrected_observations(observations=observations)
        # this returns data in the same format as above, but in the form of a dictionary of lists (metadata is not passed back):
            # eg, an input such as:
                # [[(0, (True, None, None)), (1, (False, None, None))], [(100, (True, None, None)), (200, (False, None, None))]]
            # might return an object like:
                # {0: [(0,), (1,)], 5: [(100,), (200,)]}
        # note that the dictionary key is the tracked id of the relevant object

        # use the corrected output in some way
        output_image_corrected = self.get_corrected_output_image(input_image, corrected_observations)

        # this is just to have equivalently sized images
        output_image_default = cv2.resize(output_image_default, (output_image_corrected.shape[1], output_image_corrected.shape[0]))

        # write images
        self.make_images(output_image_default, output_image_corrected)
        if self.output_path is not None:
            self.write_images(input_image, output_image_default, output_image_corrected, frame_count)

    def get_image(self):
        """

        :return:
        """
        return next(self.input_data)

    def get_observations(self, input_image):
        """
        Calls the model of your choice and generates observations.
        :param input_image:
        :return:
        """
        output_image_default, observations = self.model.calculate(input_image, suppress=True)
        return output_image_default, self.transform_observations(observations)

    def get_corrected_output_image(self, input_image, corrected_observations):
        """

        :param input_image:
        :param corrected_observations:
        :return:
        """
        if self.model_name in ['coco', 'body_25']:
            transformed = [[int(el[0]) if el[0] is not None else None for el in obj] for obj in corrected_observations.values()]

            corrected_observations = list(zip(transformed[0][::2], transformed[0][1::2]))
        return self.model.draw_results(input_image, corrected_observations)

    def do_exploratory_work(self):
        """
        This is mostly to keep in mind what the observations actually correspond to.
        :return:
        """
        if self.model_name == 'yolo':
            image = self.get_image()

            #this is output from the model
            self.feature_dict = {
                'x_min': 0,
                'x_max': 1,
                'y_min': 2,
                'y_max': 3,
                'x_centre': 4,
                'y_centre': 5,
                'score': 6
            }
            #this is an engineered additional feature
            self.feature_dict['area'] = max(self.feature_dict.values()) + 1

            self.feature_metadata = {
                'x_min': (False, 0.0, float(image.shape[1])),
                'x_max': (False, 0.0, float(image.shape[1])),
                'y_min': (False, 0.0, float(image.shape[0])),
                'y_max': (False, 0.0, float(image.shape[0])),
                'x_centre': (True, 0.0, float(image.shape[1])),
                'y_centre': (True, 0.0, float(image.shape[0])),
                'score': (False, 0.0, 1.0),
                'area': (True, 0.0, float(np.prod(image.shape)))
            }
        elif self.model_name in ['coco', 'body_25']:
            self.feature_dict = {
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

            self.feature_metadata = {}

    def engineer_features(self, observed_objects):
        """
        Append engineered features to the observation list.
        :param observed_objects:
        :return:
        """
        if self.model_name == 'yolo':
            for obj_feature_list in observed_objects:
                obj_feature_list.append((obj_feature_list[self.feature_dict['x_max']] -\
                                        obj_feature_list[self.feature_dict['x_min']]) *\
                                        (obj_feature_list[self.feature_dict['y_max']] -\
                                        obj_feature_list[self.feature_dict['y_min']]))

        return observed_objects

    def write_metadata(self, observed_objects):
        """
        Converts the observations to the correct format and appends the metadata.
        :param observed_objects:
        :return:
        """
        if self.model_name =='yolo':
            obs_with_meta = []
            for i, obj_feature_list in enumerate(observed_objects):
                obs_with_meta.append([])
                for obs, meta in zip(obj_feature_list, self.feature_metadata.values()):
                    obs_with_meta[i].append((obs, meta))
            return obs_with_meta
        else:#if self.model_name in ['coco', 'body_25']:
            # this is equivalent to empty metadata
            return [[(el, (True, None, None)) for el in obj] for obj in observed_objects]

    @staticmethod
    def transform_observations(observed_objects):
        if len(observed_objects) == 0:
            return []
        else:
            return [[observation[0] for observation in obj] for obj in observed_objects]

    def make_images(self, output_image_default, output_image_corrected):
        cv2.imshow(self.model_name, output_image_default)
        cv2.imshow('edited', output_image_corrected)

    def write_images(self, input_image, output_image_default, output_image_corrected, frame_count):
        cv2.imwrite(os.path.join(self.output_path, 'original') + str(frame_count) + '.jpg', input_image)
        cv2.imwrite(os.path.join(self.output_path, 'default') + str(frame_count) + '.jpg', output_image_default)
        cv2.imwrite(os.path.join(self.output_path, 'corrected') + str(frame_count) + '.jpg', output_image_corrected)
