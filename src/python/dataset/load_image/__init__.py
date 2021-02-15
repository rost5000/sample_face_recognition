from cv2 import COLOR_BGR2RGB
from cv2 import VideoCapture
from cv2 import cvtColor
from cv2 import imread


# This utils return RGB image from files or video

def load_from_camera(cap: VideoCapture):
    _, frame = cap.read()
    # Our operations on the frame come here
    return frame


def load_from_image(file_name: str):
    frame_test = imread('../images/Chris Hemsworth.jpg')
    gray_test = cvtColor(frame_test, COLOR_BGR2RGB)
    return gray_test
