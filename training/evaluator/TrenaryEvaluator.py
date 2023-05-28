
from sklearn.metrics import accuracy_score
import numpy as np

from utils.utils import check_tensor_equal


class TrenaryEvaluator():
    def __init__(self, ground_truths, predictions):
        self.ground_truths = ground_truths
        self.predictions = predictions

    def accuracy_score(self):
        return self._convert_to_accuracy(accuracy_score(self.ground_truths, self.predictions))

    def trend_down_accuracy(self):
        total_cnt = 0
        prediction_cnt = 0
        for i in range(len(self.ground_truths)):
            ground_truth = self.ground_truths[i]
            if self._is_trend_down_vector(ground_truth):
                prediction = self.predictions[i]
                total_cnt += 1
                if check_tensor_equal(ground_truth, prediction):
                    prediction_cnt += 1
        return self._convert_to_accuracy(prediction_cnt / total_cnt)

    def trend_up_accuracy(self):
        total_cnt = 0
        prediction_cnt = 0
        for i in range(len(self.ground_truths)):
            ground_truth = self.ground_truths[i]
            if self._is_trend_up_vector(ground_truth):
                prediction = self.predictions[i]
                total_cnt += 1
                if check_tensor_equal(ground_truth, prediction):
                    prediction_cnt += 1
        return self._convert_to_accuracy(prediction_cnt / total_cnt)

    def trend_up_or_down_accuracy(self):
        total_cnt = 0
        prediction_cnt = 0
        for i in range(len(self.ground_truths)):
            ground_truth = self.ground_truths[i]
            if not self._is_trend_even_vector(ground_truth):
                prediction = self.predictions[i]
                total_cnt += 1
                if check_tensor_equal(ground_truth, prediction):
                    prediction_cnt += 1
        return self._convert_to_accuracy(prediction_cnt / total_cnt)

    def _is_trend_down_vector(self, vector):
        return check_tensor_equal(vector, [1, 0, 0])

    def _is_trend_even_vector(self, vector):
        return check_tensor_equal(vector, [0, 1, 0])

    def _is_trend_up_vector(self, vector):
        return check_tensor_equal(vector, [0, 0, 1])

    def _convert_to_accuracy(self, prob):
        return np.round(prob * 100, 2)
