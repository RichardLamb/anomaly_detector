import numpy as np
import pandas as pd

from Entity import Entity


class Overseer:
    """
    This is the administration object for observation correction and entity tracking.
    """
    def __init__(self):
        self.n_observed_entities = 0
        self.tracked_objects = {}

    def create_entities(self, unused_observation_ids, observations):
        for unused_observation_id in unused_observation_ids:
            self.tracked_objects[self.n_observed_entities] = Entity(initial_observations=observations[unused_observation_id])
            self.n_observed_entities += 1

    def destroy_entities(self, object_ids):
        for object_id in object_ids:
            del(self.tracked_objects[object_id])

    def figure_out_observation_ownership(self, observations):
        q_factors = []

        for obj_index, object_id in enumerate(self.tracked_objects.keys()):
            q_factors.append([])
            for obs in observations:
                self.tracked_objects[object_id].add_observations(observation_list=obs, assessment_flag=False)
                q_factors[obj_index].append(self.tracked_objects[object_id].get_q_factor())

        df = pd.DataFrame(q_factors, index=self.tracked_objects.keys(), columns=range(len(observations)))
        while np.all(df.shape):
            the_minimum = np.argwhere(np.asarray(df)==df.min().min())[0]
            object_id = df.index.values[the_minimum[0]]
            obs_index = df.columns.values[the_minimum[1]]
            self.tracked_objects[object_id].add_observations(observation_list=observations[obs_index], assessment_flag=True)
            df = df.drop(index=object_id, columns=obs_index)

        if np.any(df.shape):
            if df.shape[0] != 0:
                self.destroy_entities(df.index.values)
            if df.shape[1] != 0:
                self.create_entities(df.columns.values, observations)

        # this ensures the correct q_factor is used
        for obj in self.tracked_objects.values():
            obj.get_q_factor()

        for object_id in self.tracked_objects.keys():
            self.tracked_objects[object_id].accept_new_observations()

    def add_observations(self, target, observations):
        self.tracked_objects[target].add_observations(observations)

    def accept_observations(self):
        if len(self.tracked_objects) != 0:
            self.tracked_objects[0].accept_new_observations()

    def get_corrected_observations(self, observations):
        self.figure_out_observation_ownership(observations=observations)
        return {k: [(el,) for el in obj.get_latest_correct_observations()] for k, obj in self.tracked_objects.items()}
