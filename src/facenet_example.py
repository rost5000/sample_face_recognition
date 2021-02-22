import os, shutil

import cv2  # opencv
import numpy as np
import torch

from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from pytorch_metric_learning.losses import AngularLoss, BaseMetricLossFunction
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from scipy.spatial import distance
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()

scaller = StandardScaler()
model = FaceNet().model
detector = MTCNN() # Медленно работает TODO: выпилить
face_cascade=cv2.CascadeClassifier('../cascade/haarcascade_frontalface_default.xml')
dest_path = '../dataset/img_align_sorted_light'

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output
class CustomScaller:
    def fit_transform(self, x:np.ndarray):
        # Не определился ещё
        # mean = np.mean(x, keepdims=True)
        # std = np.std(x, keepdims=True)
        # return (x - mean) / std
        max = np.max(x, keepdims=True)
        min = np.min(x, keepdims=True)
        return (x - min) / (max - min)

def norm(x, scaller=None):
    if scaller is None:
        scaller = CustomScaller()
    return scaller.fit_transform(x)

def get_embeding(x:np.ndarray):
    return model.predict(
        norm(cv2.resize(x, (160, 160))
             .reshape(-1,160,160,3))
    )

def predict_embedded_eucledian_distance(test_x, train_x, train_y):
    res = []
    for emb_test in test_x:
        min = 100
        res.append(None)
        for indx, emb_train in enumerate(train_x):
            min_new = distance.euclidean(emb_train, emb_test)
            if min_new < 0.75 and min > min_new:
                min = min_new
                res[-1] = train_y[indx]
    return res

if __name__ == "__main__":
    kf = KFold(n_splits=2)
    test_y, test_x, train_x, train_y = [[], [], [], []]
    for clss in os.listdir(dest_path):
        print("class: " + clss, end='\r')
        dates = os.listdir(dest_path + '/' + clss)
        train_index, test_index = kf.split(dates)
        for indx in train_index[0]:
            train_y.append(clss)
            frame = cv2.imread(dest_path + '/' + clss + '/' + dates[indx])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            train_x.append(frame)
        for indx in test_index[0]:
            test_y.append(clss)
            frame = cv2.imread(dest_path + '/' + clss + '/' + dates[indx])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            test_x.append(frame)

    print()
    med_val_test = 605
    med_val_train = 588
    hight_board = 1511
    transfomed_label_test = [int(x) for x in test_y]
    transfomed_label_train = [int(x) for x in train_y]


    test_y_new = np.zeros(hight_board)
    test_y_new[med_val_test:hight_board] = 1
    train_y_new = np.zeros(hight_board)
    train_y_new[med_val_train:hight_board] = 1

    print("lets train")
    # model_svc = SVC(kernel='linear', probability=True).fit(train_x, transfomed_label_test)
    # test_y_predicted_new = model_svc.predict(test_x)

    test_x_embeded, train_x_embeded = [[], []]
    for tst in test_x:
        test_x_embeded.append(get_embeding(tst)[0])
    for trn in train_x:
        train_x_embeded.append(get_embeding(trn)[0])
    test_y_predicted_new = predict_embedded_eucledian_distance(l2_normalize(test_x_embeded),
                                                               l2_normalize(train_x_embeded[med_val_train:hight_board]),
                                                               train_y_new[med_val_train:hight_board])
    for indx in range(len(test_y_predicted_new)):
        if test_y_predicted_new[indx] is None:
            test_y_predicted_new[indx] = 0


    print("Results:")
    print("Accuracy:", end=' ')
    print(np.mean([test_y_predicted_new[indx] == test_y_new[indx] for indx in range(hight_board)]))


    print("True Positive:", end=' ')
    print(np.mean([test_y_predicted_new[indx] == test_y_new[indx] for indx in range(med_val_test)]))
    print("True Negative:", end=' ')
    print(np.mean([test_y_predicted_new[indx] == test_y_new[indx] for indx in range(med_val_test, hight_board)]))
    print("False Positive:", end=' ')
    print(np.mean([test_y_predicted_new[indx] == 1 for indx in range(med_val_test)]))
    print("False Negative:", end=' ')
    print(np.mean([test_y_predicted_new[indx] == 0 for indx in range(med_val_test, hight_board)]))


