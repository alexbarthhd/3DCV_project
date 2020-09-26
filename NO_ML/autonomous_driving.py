import cv2
import time
import numpy as np

from driving_functions import config_pwm, steering, motor_ctrl
from lane_detection import Video, get_laneangle


def steering_warmup():
    pwm = config_pwm(hz=60)
    # max right:
    steering(25, pwm)
    time.sleep(3)
    steering(0, pwm)


def get_desired_direction(left_lane, right_lane, frame_width, frame_height):
    if left_lane.size != 0 and right_lane.size != 0:
        x1 = 0.5 * (right_lane[0, 0] + left_lane[0, 2])
        y1 = 0.5 * (right_lane[0, 1] + left_lane[0, 3])
    elif left_lane.size != 0:
        x1 = left_lane[0, 2] + 0.25 * frame_width
        y1 = left_lane[0, 3]
    elif right_lane.size != 0:
        x1 = right_lane[0, 0] - 0.25 * frame_width
        y1 = right_lane[0, 1]

    x2 = 0.5 * frame_width
    y2 = frame_height

    return np.array([[x1, y1, x2, y2]], dtype=np.int32)


def get_steeringangle(direction):
    ''' helper func to calc steeringangle in degrees '''
    x1, y1, x2, y2 = direction[0]
    m = (y2 - y1) / (x2 - x1)
    angle = np.arctan(m)

    return np.degrees(angle)


def main():
    pwm = config_pwm(hz=60)
    video = Video(0, 352, 288)

    while True:
        frame, frame_lines, roi_frame, left_lane, right_lane = video.get_frame()
        direction = get_desired_direction(left_lane, right_lane, 352, 288)
        steeringangle = get_steeringangle(direction)

        x1, y1, x2, y2 = direction[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(frame, f"steeringangle: {steeringangle}", (0, 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        cv2.imshow("frame", frame)
        cv2.imshow("frame w/ lines", frame_lines)
        cv2.imshow("ROI frame", roi_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pass

if __name__ == "__main__":
    main()
