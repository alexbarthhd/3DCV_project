import cv2
import numpy as np

from dataclasses import dataclass, field


def lane_detection(func):
    ''' decorator to detect lanes in video frame '''
    def func_wrapper(*args, **kwargs):
        frame = func(*args, **kwargs)

        # TODO: detect ROI
        black = np.zeros((288, 352, 1), dtype=np.uint8)
        roi = np.array([[0, 288], [0, 230], [88, 130], [264, 130], [352, 230],
                        [352, 288]])

        stencil = cv2.fillConvexPoly(black, roi, 1)
        frame = cv2.bitwise_and(frame, frame, mask=stencil)

        # TODO: to grayscale and tresholding
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_binary = cv2.threshold(frame_gray, 80, 255, cv2.THRESH_BINARY)[1]

        # TODO: Hough line transformation

        return frame_binary, frame

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
        frame, stencil = video.get_frame()
        cv2.imshow("frame", frame)
        cv2.imshow("stencil", stencil)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
