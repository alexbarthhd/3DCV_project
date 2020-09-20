import cv2
import numpy as np

from dataclasses import dataclass, field


def lane_detection(func):
    ''' decorator to detect lanes in video frame '''
    def func_wrapper(*args, **kwargs):
        frame = func(*args, **kwargs)

        # TODO: detect ROI

        # TODO: to grayscale and tresholding
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, frame_tresh = cv2.threshold(frame_gray, 127, 255, cv2.THRESH_BINARY)

        # TODO: Hough line transformation

        return frame_tresh

    return func_wrapper


@dataclass
class Video:
    '''
    Dataclass to manage video input with
    src = 0 -> webcam
    '''
    src: int
    width: float
    height: float
    cap: object = field(init=False)

    def __post_init__(self):
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

    @lane_detection
    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame


if __name__ == "__main__":
    video = Video(0, 352, 288)

    while True:
        frame = video.get_frame()
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
