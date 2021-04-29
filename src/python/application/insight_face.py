import time

import cv2
import numpy as np
import torch
from PIL import Image

import src.python.dataset.load_image as li
from src.python.insight_face.Learner import face_learner
from src.python.insight_face.config import get_config
from src.python.insight_face.mtcnn import MTCNN
from src.python.insight_face.utils import get_emb_from_frame


class Insight_Face:
    def __init__(self):
        self.conf = get_config(False)
        self.mtcnn = MTCNN()

        self.learner = face_learner(self.conf)
        self.learner.threshold = 0.8
        self.learner.load_state(self.conf, 'cpu_final.pth', True, True)
        self.learner.model.eval()

        self._boxes = [[]]
        self._landmarks = [[]]
        self._image = None
        self._faces = None

    @property
    def boxes(self):
        return self._boxes

    @boxes.setter
    def boxes(self, value):
        self._boxes = value if len(value) else [[]]

    @property
    def landmarks(self):
        return self._landmarks

    @landmarks.setter
    def landmarks(self, value):
        self._landmarks = value if len(value) else [[]]

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, value):
        self._faces = value

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        """Set image from frame"""

        self._image = Image.fromarray(cv2.cvtColor(value, cv2.COLOR_BGR2RGB))

    def train_mvp(self):
        embs = []
        embeddings = []
        cap = cv2.VideoCapture(0)
        while len(embs) < 10:
            frame = li.load_from_camera(cap)
            emb = get_emb_from_frame(self.conf, self.learner.model, self.mtcnn, frame, tta=False)
            if emb is not None:
                embs.append(emb)

        embeddings.append(torch.cat(embs).mean(0, keepdim=True))
        embeddings = torch.cat(embeddings)
        names = np.array(["Unknown", "Aleksey"])
        return embeddings, names

    def infer(self, embeddings, names, boxes, faces, frame):
        if faces is None:
            print('No bounding boxes')
            return
        begin_detector = time.time()
        results, dists = self.learner.infer(self.conf, faces, embeddings, False)
        end_detector = time.time()
        print(f"Время infer: {end_detector - begin_detector}")
        # тут завязались на то, что первый элемент names обязательно Unknown
        predictions = [names[ind + 1] for ind in results]
        for ind, box in enumerate(boxes):
            print(f'predict={predictions[ind]}, dist = {dists[ind]}')

            x_l_face_mtcnn, y_up_face_mtcnn, x_r_face_mtcnn, y_down_face_mtcnn = round(box[0]), round(
                box[1]), round(box[2]), round(box[3])

            cv2.putText(
                frame,
                "ArcFace+ResNet50",
                (x_l_face_mtcnn + 10, y_up_face_mtcnn - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (34, 139, 34), 2
            )
            cv2.rectangle(
                frame,
                (x_l_face_mtcnn, y_up_face_mtcnn),
                (x_r_face_mtcnn, y_down_face_mtcnn),
                (34, 139, 34) if predictions[ind] == 'Aleksey' else (0, 69, 255),
                6
            )
