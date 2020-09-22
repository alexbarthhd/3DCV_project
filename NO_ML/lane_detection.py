import cv2
import numpy as np

from dataclasses import dataclass, field


def split_lines(array):
    ''' helper func to distinguish lines marking left and right lanes '''
    left_lines, right_lines = [], []

    for _, item in enumerate(array):
        if ((item[0, 0] == 0) and (item[0, 1] > item[0, 3])):
            left_lines.append(item)
        else:
            right_lines.append(item)

    return np.array(left_lines), np.array(right_lines)


def lane_detection(func):
    ''' decorator to detect lanes in video frame '''
    def func_wrapper(*args, **kwargs):
        frame = func(*args, **kwargs)
        lanes = []

        # TODO: detect ROI
        white = np.ones((288, 352, 1), dtype=np.uint8) * 255
        roi = np.array([[0, 288], [0, 230], [88, 130], [264, 130], [352, 230],
                        [352, 288]])

        stencil = cv2.fillConvexPoly(white, roi, 0)

        # TODO: to grayscale and tresholding
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_binary = cv2.threshold(frame_gray, 80, 255, cv2.THRESH_BINARY)[1]
        roi_frame = cv2.add(frame_binary, stencil)

        # TODO: Hough line transformation
        lines = cv2.HoughLinesP(cv2.bitwise_not(roi_frame), 1, np.pi/180, 30,
                                minLineLength=80, maxLineGap=200)

        # get lanes
        left_lines, right_lines = split_lines(lines)

        if left_lines.size != 0:
            left_lane = np.mean(left_lines, axis=0, dtype=np.int32)
            x1, y1, x2, y2 = left_lane[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)
            lanes.append(left_lane)

        if right_lines.size != 0:
            right_lane = np.mean(right_lines, axis=0, dtype=np.int32)
            x1, y1, x2, y2 = right_lane[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
            lanes.append(right_lane)

        return frame, lanes

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
        frame, lanes = video.get_frame()
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
