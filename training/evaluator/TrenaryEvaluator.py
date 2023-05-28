
from sklearn.metrics import accuracy_score
import numpy as np


class TrenaryEvaluator():
    def __init__(self, ground_truths, predictions):
        self.ground_truths = np.argmax(ground_truths, axis=1)
        self.predictions = np.argmax(predictions, axis=1)

    def accuracy_score(self):
        return self._convert_to_accuracy(accuracy_score(self.ground_truths, self.predictions))

    def trend_down_accuracy(self):
        indices = np.where(self.predictions == 0)[0]
        truths = self.ground_truths[indices]
        preds = self.predictions[indices]
        return self._convert_to_accuracy(accuracy_score(
            truths, preds))

    def trend_up_accuracy(self):
        indices = np.where(self.predictions == 2)[0]
        truths = self.ground_truths[indices]
        preds = self.predictions[indices]
        return self._convert_to_accuracy(accuracy_score(
            truths, preds))

    def trend_up_or_down_accuracy(self):
        indices = np.where(self.predictions != 1)[0]
        truths = self.ground_truths[indices]
        preds = self.predictions[indices]
        return self._convert_to_accuracy(accuracy_score(
            truths, preds))

    def _convert_to_accuracy(self, prob):
        return np.round(prob * 100, 2)
