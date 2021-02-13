import ctypes  # An included library with Python install.

import cv2
import numpy as np

import src.python.dataset.load_image as li
import src.python.facenet as  fs
import src.python.facenet.utils as face_utils

cap = cv2.VideoCapture(0)
emb_train = []
y_train = []

if __name__ == "__main__":
    while True:
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
            break

    code = ctypes.windll.user32.MessageBoxW(0, "Your Authentetification", "Your Authentetification", 1)

    if code == 1:
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame = li.load_from_camera(cap)
            faces_coord = face_utils.get_face_from_image(frame)
            print(faces_coord)
            if faces_coord.__len__() == 1:
                x_face, y_face, w_face, h_face = [faces_coord[0][0], faces_coord[0][1], faces_coord[0][2],
                                                  faces_coord[0][3]]
                emb_test = fs.get_embeding(
                    frame[y_face:y_face + h_face, x_face:x_face + w_face]
                )
                res = fs.predict_embedded_euclidean_distance(np.array([emb_test]), np.array(emb_train), y_train)
                cv2.rectangle(
                    frame,
                    (faces_coord[0][0], faces_coord[0][1]),
                    (faces_coord[0][0] + faces_coord[0][2], faces_coord[0][1] + faces_coord[0][3]),
                    (0, 0, 255) if res[0] is None else (255, 0, 0),
                    6
                )
            cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()
