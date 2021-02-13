import numpy as np


class CustomScaller:
    def fit_transform(self,x: np.ndarray):
        max = np.max(x, keepdims=True)
        min = np.min(x, keepdims=True)
        return (x - min) / (max - min)
