from cv2 import COLOR_BGR2GRAY
from cv2 import cvtColor


def image_process_from_cv2(frame):
    gray = cvtColor(frame, COLOR_BGR2GRAY)
    return gray
