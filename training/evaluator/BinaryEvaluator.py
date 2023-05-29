
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np


class BinaryEvaluator():
    def __init__(self, ground_truths, predictions, threshold):
        self.ground_truths = np.array(ground_truths)
        predictions = np.array(predictions)
        self.predictions = (predictions > threshold).astype(int)

    def accuracy_score(self, mode):
        acc = self._convert_to_accuracy(
            accuracy_score(self.ground_truths, self.predictions))
        print(f"{mode} overall accuracy: {acc}%")

    def get_positive_accuracy_score(self):
        indices = np.where(self.predictions == 1)[0]
        truths = self.ground_truths[indices]
        preds = self.predictions[indices]
        try:
            accuracy = self._convert_to_accuracy(accuracy_score(
                truths, preds))
            return accuracy, len(indices)
        except:
            return 0.0, 0.0

    def report_positive_accuracy_score(self, mode):
        indices = np.where(self.predictions == 1)[0]
        truths = self.ground_truths[indices]
        preds = self.predictions[indices]
        try:
            accuracy = self._convert_to_accuracy(accuracy_score(
                truths, preds))
            print(f"Out of {len(indices)} predictions:")
            print(f"{mode} positive accuracy with current model: {accuracy}%")
        except:
            pass

    def negative_accuracy_score(self, mode):
        indices = np.where(self.predictions == 0)[0]
        truths = self.ground_truths[indices]
        preds = self.predictions[indices]
        try:
            accuracy = self._convert_to_accuracy(accuracy_score(
                truths, preds))
            print(f"Out of {len(indices)} predictions:")
            print(f"{mode} negative accuracy with current model: {accuracy}%")
        except:
            pass

    def f1_score(self, mode):
        accuracy = self._convert_to_accuracy(
            f1_score(self.ground_truths, self.predictions))
        print(f"{mode} F1 score: {accuracy}%")

    def _convert_to_accuracy(self, prob):
        return np.round(prob * 100, 2)
