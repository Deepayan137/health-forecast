import pdb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def _bias(self):
        return np.mean(self.y_true - self.y_pred)

    def _relative_bias(self):
        return self._bias() / np.mean(self.y_true)

    def _mse(self):
        return np.mean((self.y_true - self.y_pred)**2)
    
    def _rmse(self):
        return np.sqrt(self._mse())

    def _rrmse(self):
        return self._rmse() / np.mean(self.y_true)

    def __call__(self):
        return {
            "bias": self._bias(),
            "relative_bias": self._relative_bias(),
            "mse": self._mse(),
            "rmse": self._rmse(),
            "rrmse": self._rrmse()
        }
