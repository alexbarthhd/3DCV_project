import cv2
import time
import numpy as np

from driving_functions import config_pwm, steering, motor_ctrl
from lane_detection import Video, get_laneangle


if __name__ == "__main__":
    pwm = config_pwm(hz=60)
    video = Video(0, 352, 288)
    steeringangle = 0
    left_laneangle = 0
    right_laneangle = 0

    while True:
        frame, roi_frame, lanes = video.get_frame()
        left_laneangle = get_laneangle(lanes[0])

        if len(lanes) > 1:
            right_laneangle = get_laneangle(lanes[1])

        if left_laneangle < -50:
            steeringangle = -45 - left_laneangle

        if right_laneangle > 50:
            steeringangle = -(right_laneangle - 45)

        #steering(steeringangle, pwm)
        print(steeringangle)
        time.sleep(1)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    '''
    # max right:
    steering(25, pwm)
    time.sleep(3)

    steering(0, pwm)
    '''
