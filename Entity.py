import numpy as np

from Historian import Historian


class Entity:
    """This class administers the series of Historian objects that are relevant to a given tracked Entity."""
    def __init__(self, initial_observations):
        self.historians = [Historian(observation) for observation in initial_observations + [(0.0, (False, None, None))]]
        self.observation_count = 0

    def add_observations(self, observation_list, assessment_flag):
        for historian, observation in zip(self.historians, observation_list):
            historian.add_observation(observation, assessment_flag=assessment_flag)

    def accept_new_observations(self):
        for historian in self.historians:
            historian.accept_new_observation()
        self.observation_count += 1

    def get_q_factor(self):
        """

        :return: description=q_factor defines how close the newest observations are to the previous ones using some metric
                 note=lower is intended to be better
                 type=float
                 eg=0.321
        """
        q_factor = np.linalg.norm([historian.get_q_factor() for historian in self.historians], 2)
        self.historians[-1].add_observation((q_factor,), assessment_flag=False)
        return q_factor

    def get_latest_correct_observations(self):
        return [historian.time_series[-1] for historian in self.historians]
