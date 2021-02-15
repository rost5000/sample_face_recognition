from cv2 import CascadeClassifier
from mtcnn.mtcnn import MTCNN

face_cascade = CascadeClassifier('../../resources/cascade/haarcascade_frontalface_default.xml')
detector = MTCNN()  # Медленно работает TODO: выпилить


def get_face_from_image(frame):
    """
    This is an example to  use that:

    face_test_coord = get_face_from_image(frame)
    for (x_face,y_face,w_face,h_face) in face_test_coord:
        plt.imshow(frame[y_face:y_face+h_face, x_face:x_face+w_face])

    :param frame: the rgb frame for image
    :return: coordinates
    """
    # face_coord = [face['box'] for face in detector.detect_faces(frame)]
    face_coord = face_cascade.detectMultiScale(frame,
                                scaleFactor=1.1,
                                minNeighbors=8)
    return face_coord
