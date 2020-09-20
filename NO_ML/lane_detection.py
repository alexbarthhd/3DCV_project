import cv2
import numpy as np

from dataclasses import dataclass, field


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

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame


if __name__ == "__main__":
    video = Video(cv2.CAP_DSHOW, 352, 288)

    while True:
        frame = video.get_frame()
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
