import cv2
import numpy as np


if __name__ == "__main__":
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

    # TODO: Hough line transformation
    lines = cv2.HoughLinesP(cv2.bitwise_not(roi_frame), 1, np.pi/180, 30, maxLineGap=200)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
