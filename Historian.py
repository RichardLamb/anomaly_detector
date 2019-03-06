import warnings

import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

from conf import HISTORIAN_MEMORY_DURATION
from conf import ANOMALY_DETECTION_THRESHOLD__N_STDS
from conf import TIME_SERIES_TRAIN_WINDOW as PREDICTION_WINDOW
from conf import HISTORIAN_NONE_MEMORY_MAXIMUM
from conf import HISTORIAN_PREDICTION_STRATEGY

from conf import SGOLAY_POLYNOMIAL_ORDER

from conf import observation_metadata_indices

# to stop scipy from crying about something we cant change (sgolay)
warnings.simplefilter(action='ignore', category=FutureWarning)

class Historian:
    """This class administers time-series related functionality for a given observation stream."""
    def __init__(self, initial_observation):
        """

        :param initial_observation:
        """
        self.add_observation(observation=initial_observation, assessment_flag=False)
        self.unpack_observation_metadata(initial_observation)
        self.predictor = HISTORIAN_PREDICTION_STRATEGY
        self.prediction_window = PREDICTION_WINDOW

        self.obs_history = [None] * HISTORIAN_MEMORY_DURATION
        self.obs_history[-1] = initial_observation[0]

        self.time_series = [None] * HISTORIAN_MEMORY_DURATION
        self.time_series[-1] = initial_observation[0]

        self.anomaly_tracker = [False] * HISTORIAN_MEMORY_DURATION
        self.none_memory = 0
        self.none_memory_max = HISTORIAN_NONE_MEMORY_MAXIMUM

    def unpack_observation_metadata(self, initial_observation):
        """
        @the next person who reads this: I'm sorry about this.
        :return:
        """
        self.metadata = {}
        for key, val in observation_metadata_indices.items():
            try:
                self.metadata[key] = initial_observation[1][val]
            except IndexError:
                self.metadata[key] = None

    """
    Methods for adding and assessing observations:
    """
    def add_observation(self, observation, assessment_flag):
        """
        Accepts a new observation and tests it.
        :param observation: description=a tuple containing a new observation and a tuple containing observation metadata
                            notes=the actual observation value is observation [0], the metadata is observation[1]
                            type=tuple
                            size=2
                            example=(42.42, (True, None, False))
        :param assessment_flag: description=flags whether the observation should also be assessed
                                type=bool
                                example=True
        :return:
        """
        self.observation = observation[0]

        if assessment_flag and self.observation is not None and self.time_series[-1] is not None:
            self.latest_obs_is_anomaly = self._test_observation()
        else:
            self.latest_obs_is_anomaly = False

    def _test_observation(self):
        """
        Checks to see if the most recently added observation is anomalous using the _detect_anomalies method.
        :return: description=anomaly flag
                 type=bool
                 example=False
        """
        anomalies = self._detect_anomalies(np.asarray(self.obs_history + [self.observation]), threshold=ANOMALY_DETECTION_THRESHOLD__N_STDS)
        return anomalies[-1]  # == True

    @staticmethod
    def _detect_anomalies(arr, threshold, suppress=True):
        """The function recieves a clean array and detects the anomalies bigger than threshold (units of std)"""
        arr_diff = np.diff(np.diff(arr[arr != None]))
        if len(arr_diff) == 0:
            return [False]
        n_anomalies = 0
        # define anomalies as a list of False
        anomalies = np.asarray(len(arr_diff) * [False])

        while True:
            current_std = np.std(arr_diff[~anomalies])
            # if current_std == 0:
            #     break
            anomalies = np.abs(arr_diff) >= (threshold * current_std)

            if anomalies[-1] == True or np.count_nonzero(anomalies) == n_anomalies:
                break
            else:
                n_anomalies = np.count_nonzero(anomalies)

        if not suppress:
            import matplotlib.pyplot as plt
            plt.subplot(1, 2, 1)
            plt.plot(range(len(arr_diff)), arr_diff, color='blue')
            if n_anomalies is not 0:
                plt.scatter(np.argwhere(anomalies), arr_diff[anomalies], color='red')

            plt.subplot(1, 2, 2)
            ano = [False] + anomalies.tolist() + [False]
            plt.plot(range(len(arr)), arr, color='blue')
            plt.scatter(np.argwhere(ano), arr[ano], color='red')
        return anomalies

    """
    Methods for accepting observations:
    """
    def accept_new_observation(self):
        """

        :return:
        """
        self.obs_history.append(self.observation)
        del(self.obs_history[0])

        self.time_series.append(self._get_corrected_observation())
        del(self.time_series[0])

        self.anomaly_tracker.append(self.latest_obs_is_anomaly)
        del(self.anomaly_tracker[0])

        self._predict_next()

    def _get_corrected_observation(self):
        """
        Uses some arbitrary logic to acquire the corrected observation depending on whether the value is regarded as an anomaly.
        :return: description=a corrected observation
                 type=float
                 example=42.42
        """
        if not self.latest_obs_is_anomaly:
            if self.time_series[-1] is None and self.observation is None:
                self.none_memory = 0
                return None
            else:
                relevant_values = np.asarray([self.time_series[-1], self.observation])
                if np.count_nonzero(relevant_values != None) == 1:
                    self.none_memory += 1
                    if self.none_memory == self.none_memory_max:
                        self.none_memory = 0
                        return None
        else:
            relevant_values = np.asarray((self.observation, self.time_series[-1], self.prediction))

        return np.mean(relevant_values[relevant_values != None])

    def _predict_next(self):
        """
        Pseudo-wrapper for writing the next predicted value. This is mostly for convenience in editing the model.
        :return:
        """
        self.prediction = self._deploy_model(self.obs_history, PREDICTION_WINDOW)

    def _deploy_model(self, previous_data, prediction_window):
        """the function receives data up to this point and a prediction window that defines hoe many frames to train on
        and predicts what is the next point going to be"""
        the_data = np.asarray(previous_data[-prediction_window:])

        if self.predictor == 'sgolay_filter':
            y_data, x_data = self.interp_to_clean(the_data)

            if len(y_data) % 2 != 0:
                y_data = y_data[1:]
                x_data = x_data[1:]
            if len(y_data) < SGOLAY_POLYNOMIAL_ORDER:
                return None
            predicted_point = self.sgolay_filt(x_data, y_data)
        else:
            y_data, x_data = self.remove_to_clean(the_data)
            if np.count_nonzero(the_data != None) < prediction_window/2:
                return None

            if self.predictor == 'linear':
                predicted_point = self.linear(np.append(x_data, x_data[-1] + 1), y_data)
            else:
                predicted_point = self.fake_AR(np.append(x_data, x_data[-1] + 1), y_data)

        return predicted_point

    def remove_to_clean(self, the_data):
        """
        Cleans the data by removing None types.
        :param the_data:
        :return:
        """
        x_data = np.arange(len(the_data))[the_data != None]
        return the_data[the_data != None], x_data

    def interp_to_clean(self, the_data):
        """
        Cleans None types by linearly interpolating values.
        :param the_data:
        :return:
        """
        while len(the_data) != 0 and the_data[0] == None:
            the_data = the_data[1:]
        front_edge = None
        for i in range(1, len(the_data)):
            if the_data[i] is None:
                if front_edge is None:
                    front_edge = i
            else:
                if front_edge is not None:
                    the_data[front_edge:i] = np.linspace(the_data[front_edge - 1], the_data[i], i - front_edge + 2)[
                                             1:-1]
                    front_edge = None
        return the_data[the_data != None], np.arange(len(the_data[the_data != None])+1)

    def linear(self, x_data, y_data):
        reg = LinearRegression().fit(x_data[:-1].reshape(-1, 1), y_data[y_data != None].reshape(-1, 1))
        return reg.predict(x_data.reshape(-1, 1))[-1][0]

    def fake_AR(self, x_data, y_data):
        reg = LinearRegression()
        reg.fit(x_data[:-1].reshape(-1, 1), y_data.reshape(-1, 1))
        y_pred = reg.predict(x_data.reshape(-1, 1))

        stationary = y_data - y_pred[:-1].ravel()
        offset = 2

        the_vars = []
        for width in range(offset, int(len(stationary) / 2)):
            the_box = self.squash(stationary, width)
            the_vars.append(np.mean(np.var(the_box, axis=0)))

        if len(the_vars) == 0:
            return None

        optimal_width = np.asarray(the_vars).argmin() + offset
        pred = stationary[-optimal_width] + y_pred[-1]
        return pred[0]

    def sgolay_filt(self, x_data, y_data):
        reg = LinearRegression()
        reg.fit(x_data[:-1].reshape(-1, 1), y_data.reshape(-1, 1))
        y_pred = reg.predict(x_data.reshape(-1, 1))

        return savgol_filter(np.append(y_data, y_pred[-1]), len(x_data), SGOLAY_POLYNOMIAL_ORDER, deriv=0)[-1]

    @staticmethod
    def squash(the_data, width):
        """
        This is effectively numpy reshape (from 1 to 2 dimensions) with a 1 element overlap.
        :param the_data:
        :param width:
        :return:
        """
        if len(the_data) % (width - 1) == 0:
            height = int(len(the_data) / (width - 1)) - 1
        else:
            height = int(len(the_data) / (width - 1))

        the_box = np.zeros(shape=(height, width), dtype=np.float32)
        for row in range(height):
            the_box[row, :] = the_data[row * width - row:(row + 1) * width - row]
        return the_box

    def get_q_factor(self):
        """
        q_factor is a proxy for similarity between the observation and the last observed value.
        Note that q_factor is designed to be used in conjunction with the q_factors of other relevant Historian objects within an Entity tracker class.
        Higher q_factors are intended to be worse.
        :return: type=float
                 example=0.321
        """
        if self.time_series[-1] is None or self.observation is None:
            return 0.0
        if self.metadata['q factor applicability']:
            if self.metadata['value range minimum'] is not None and self.metadata['value range maximum'] is not None:
                return ((self.time_series[-1] - self.observation) - self.metadata['value range minimum']) /\
                       (self.metadata['value range maximum'] - self.metadata['value range minimum'])
            else:
                return self.time_series[-1] - self.observation
        else:
            return 0.0
