import cv2
import numpy as np

def split_lines(array):
    left_lines, right_lines = [], []

    for _, item in enumerate(array):
        if ((item[0, 0] == 0) and (item[0, 1] > item[0, 3])):
            left_lines.append(item)
        else:
            right_lines.append(item)

    return np.array(left_lines), np.array(right_lines)


def get_laneangle(lane):
    x1, y1, x2, y2 = lane[0]
    m = (y2 - y1) / (x2 - x1)
    angle = np.arctan(m)

    return np.degrees(angle)


def main():
    frame = cv2.imread("lane_frame.jpg")
    frame = cv2.resize(frame, (352, 288))

    # TODO: detect ROI
    white = np.ones((288, 352, 1), dtype=np.uint8) * 255
    roi = np.array([[0, 288], [0, 230], [88, 130], [264, 130], [352, 230],
                    [352, 288]])

    stencil = cv2.fillConvexPoly(white, roi, 0)

    # TODO: to grayscale and tresholding
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_binary = cv2.threshold(frame_gray, 80, 255, cv2.THRESH_BINARY)[1]
    roi_frame = cv2.add(frame_binary, stencil)

    # TODO: Probabilistic Hough line transformation
    # minLineLength to filter out lines over front wheel
    # cv2.bitwise_not(roi_frame) to invert frame to mimic the output of a canny edge-detector
    lines = cv2.HoughLinesP(cv2.bitwise_not(roi_frame), 1, np.pi/180, 30,
                            minLineLength=80, maxLineGap=200)

    # TODO: filter to get one line for each lane
    left_lines, right_lines = split_lines(lines)
    left_lane = np.mean(left_lines, axis=0, dtype=np.int32)
    right_lane = np.mean(right_lines, axis=0, dtype=np.int32)

    x1, y1, x2, y2 = left_lane[0]
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)

    x1, y1, x2, y2 = right_lane[0]
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)

    left_laneangle = get_laneangle(left_lane)
    right_laneangle = get_laneangle(right_lane)

    print(left_laneangle, right_laneangle)

    cv2.imshow("frame", frame)
    cv2.imshow("stencil", stencil)
    cv2.imshow("roi_frame", roi_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
