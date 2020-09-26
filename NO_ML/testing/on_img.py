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


def split_left_right(array, frame_width, frame_height):
    left_lines, right_lines = [], []

    for _, item in enumerate(array):
        if (0 <= item[0, 0] <= (1/5) * frame_width) and \
           (0 <= item[0, 2] <= (2/3) * frame_width):
            left_lines.append(item)

        elif ((1/3) * frame_width <= item[0, 0] <= frame_width) and \
             ((3/5) * frame_width <= item[0, 2] <= frame_width):
            right_lines.append(item)

    return np.array(left_lines), np.array(right_lines)


def lines2lane_fast(lines):

    return lane


def get_desired_direction(left_lane, right_lane, frame_width, frame_height):
    if left_lane.size != 0 and right_lane.size != 0:
        x1 = 0.5 * (right_lane[0, 0] + left_lane[0, 2])
        y1 = 0.5 * (right_lane[0, 1] + left_lane[0, 3])
    elif left_lane.size != 0:
        x1 = left_lane[0, 2]
        y1 = left_lane[0, 3]
    elif right_lane.size != 0:
        x1 = right_lane[0, 0]
        y1 = right_lane[0, 1]

    x2 = 0.5 * frame_width
    y2 = frame_height

    return np.array([[x1, y1, x2, y2]], dtype=np.int32)

def get_steeringangle(direction):
    ''' helper func to calc steeringangle in degrees '''
    x1, y1, x2, y2 = direction[0]

    # b/c arctan won't work for vertical directions
    if abs(x1 - x2) > 3:
        m = (y2 - y1) / (x2 - x1)
        angle = np.arctan(m)
    else:
        angle = 0.5 * np.pi


    print(f"intermediate angle: {angle}")

    # left [0°, -25°]
    if 0 <= angle < (0.5 * np.pi):
        angle = angle - 0.5 * np.pi
        if angle < -0.436:
            angle = -0.436

    # right [0°, 25°]
    elif angle < 0:
        angle = 0.5 * np.pi + angle
        if angle > 0.436:
            angle = 0.436

    # center
    else:
        angle = 0

    return np.degrees(angle)

def get_laneangle(lane):
    x1, y1, x2, y2 = lane[0]
    m = (y2 - y1) / (x2 - x1)
    angle = np.arctan(m)

    return np.degrees(angle)


def main():
    #frame = cv2.imread("frame2.png")
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
                            minLineLength=80, maxLineGap=50)


    for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # TODO: filter to get one line for each lane
    left_lines, right_lines = split_left_right(lines, 352, 288)

    # color left lines in green
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        left_lane = np.mean(left_lines, axis=0, dtype=np.int32)
        x1, y1, x2, y2 = left_lane[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)

    for line in right_lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

      right_lane = np.mean(right_lines, axis=0, dtype=np.int32)
      x1, y1, x2, y2 = right_lane[0]
      cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)

    direction = get_desired_direction(left_lane, right_lane, 352, 288)
    steeringangle = get_steeringangle(direction)
    print(f"steeringangle: {steeringangle}")
    print(f"left_lane: {left_lane}, right_lane: {right_lane}")
    print(f"direction: {direction}")

    x1, y1, x2, y2 = direction[0]
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)


    cv2.imshow("frame", frame)
    cv2.imshow("stencil", stencil)
    cv2.imshow("roi_frame", roi_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
