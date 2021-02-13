import cv2
import numpy as np
from keras_facenet import FaceNet
from scipy.spatial import distance

from .utils.CustomScaller import CustomScaller

model = FaceNet().model


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def norm(x, scaller=None):
    if scaller is None:
        scaller = CustomScaller()
    return scaller.fit_transform(x)


def get_embeding(x: np.ndarray):
    return model.predict(
        norm(cv2.resize(x, (160, 160))
             .reshape(-1, 160, 160, 3))
    ).flatten()


def predict_embedded_euclidean_distance(test_x: np.ndarray, train_x: np.ndarray, train_y: np.ndarray):
    res = []
    for emb_test in test_x:
        min = 100
        res.append(None)
        for indx, emb_train in enumerate(train_x):
            min_new = distance.euclidean(emb_train, emb_test)
            if min_new < 1 and min > min_new:
                min = min_new
                res[-1] = train_y[indx]
    return res

