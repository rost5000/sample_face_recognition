import ctypes  # An included library with Python install.
import time
import cv2
import numpy as np

import src.python.dataset.load_image as li
import src.python.facenet as fs
import src.python.facenet.utils as face_utils
from src.python.insight_face.mtcnn import MTCNN
from src.python.application.insight_face import Insight_Face
from PIL import Image

cap = cv2.VideoCapture(0)
emb_train = []
y_train = []
is_use_insight_face = True
is_use_windows_message_box = False

if __name__ == "__main__":
    count = 0
    while count < 10:
        frame = li.load_from_camera(cap)
        cv2.imshow('frame', frame)
        faces_coord = face_utils.get_face_from_image(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)
        if faces_coord.__len__() == 1:
            x_face, y_face, w_face, h_face = [faces_coord[0][0], faces_coord[0][1], faces_coord[0][2],
                                              faces_coord[0][3]]
            emb_train.append(fs.get_embeding(
                frame[y_face:y_face + h_face, x_face:x_face + w_face]
            ))
            y_train.append(1)
            count += 1

    if is_use_insight_face:
        insight_face = Insight_Face()
        mtcnn = MTCNN()
        embeddings, names = insight_face.train_mvp()

    if is_use_windows_message_box:
        code = ctypes.windll.user32.MessageBoxW(0, "Your Authentetification", "Your Authentetification", 1)
    else:
        code = 1  # на маке нет windll meassageBox

    if code == 1:
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame = li.load_from_camera(cap)

            faces_coord = face_utils.get_face_from_image(frame)
            print(f"\nFace coord by normal face detection: {faces_coord}")

            if is_use_insight_face:
                begin = time.time()
                insight_face.image = frame
                begin_detector = time.time()
                insight_face.boxes, insight_face.landmarks = mtcnn.detect_faces(insight_face.image)
                end_detector = time.time()
                print(f"Время детектора: {end_detector - begin_detector}")
                print(f'Faces coord = {insight_face.boxes}')
                if len(insight_face.boxes[0]):

                    begin_detector = time.time()
                    insight_face.faces = mtcnn.align_multi(
                        insight_face.image,
                        boxes=insight_face.boxes,
                        landmarks=insight_face.landmarks
                    )
                    end_detector = time.time()
                    print(f"Время align_multi: {end_detector - begin_detector}")
                    insight_face.infer(
                        embeddings=embeddings,
                        names=names,
                        boxes=insight_face.boxes,
                        faces=insight_face.faces,
                        frame=frame
                    )
                end = time.time()
                print(f"Время суммарное: {end - begin}")

            for face_coord in faces_coord:
                x_face, y_face, w_face, h_face = [face_coord[0], face_coord[1], face_coord[2],
                                                  face_coord[3]]
                emb_test = fs.get_embeding(
                    frame[y_face:y_face + h_face, x_face:x_face + w_face]
                )
                res = fs.predict_embedded_euclidean_distance(np.array([emb_test]), np.array(emb_train), y_train)
                cv2.putText(
                    frame,
                    "FaceNet",
                    (x_face + 10, y_face - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (34, 139, 34), 2
                )
                cv2.rectangle(
                    frame,
                    (face_coord[0], face_coord[1]),
                    (face_coord[0] + face_coord[2], face_coord[1] + face_coord[3]),
                    (34, 139, 34) if res[0] else (0, 69, 255),
                    6
                )
            # show_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # show_img = Image.fromarray(show_img)
            # show_img.show()
            cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()
