import cv2
import time
import numpy as np
import threading

from driving_functions import config_pwm, steering, motor_ctrl, go_slow_multistep
from lane_detection import Video, get_laneangle


def steering_warmup():
    ''' turn right-left-center '''
    pwm = config_pwm(hz=60)

    steering(25, pwm)
    time.sleep(2)
    steering(-25, pwm)
    time.sleep(2)
    steering(0, pwm)


def get_desired_direction(left_lane, right_lane, frame_width, frame_height):
    ''' helper func to calc desired direction based on given lanes '''
    if left_lane.size != 0 and right_lane.size != 0:
        x1 = 0.5 * (right_lane[0, 0] + left_lane[0, 2])
        y1 = 0.5 * (right_lane[0, 1] + left_lane[0, 3])
    elif left_lane.size != 0:
        x1 = left_lane[0, 2] + 0.25 * frame_width
        y1 = left_lane[0, 3]
    elif right_lane.size != 0:
        x1 = right_lane[0, 0] - 0.25 * frame_width
        y1 = right_lane[0, 1]

    # steer straight ahead if nothing is detected
    else:
        x1 = 0.5 * frame_width
        y1 = 0.5 * frame_height

    x2 = 0.5 * frame_width
    y2 = frame_height

    return np.array([[x1, y1, x2, y2]], dtype=np.int32)


def get_steeringangle(direction):
    ''' helper func to calc steeringangle in degrees [-25°, 25°] '''
    x1, y1, x2, y2 = direction[0]

    # b/c arctan won't work for vertical directions
    if abs(x1 - x2) > 3:
        m = (y2 - y1) / (x2 - x1)
        angle = np.arctan(m)
    else:
        angle = 0.5 * np.pi

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


def stabilize_steeringangle(steeringangle, last_steeringangle, max_deviation):
    ''' helper func to stabilze steeringangle by comparing it to the previous
        steeringangle '''
    deviation = steeringangle - last_steeringangle

    if abs(deviation) > max_deviation:
        if steeringangle > last_steeringangle:
            steeringangle = last_steeringangle + max_deviation
        else:
            steeringangle = last_steeringangle - max_deviation

    return steeringangle


def main(gen_dataset=False, stabilize=False, acc=18.5):
    ''' drives the car with contant speed ~ acc, can be used to generate datasets
        labels are steeringangles and stored as last part of the img-names '''
    pwm = config_pwm(hz=60)
    video = Video(0, 352, 288)
    last_steeringangle = 0

    # start motor
    motor_ctrl(acc, pwm)

    try:
        while True:
            frame, frame_lines, roi_frame, left_lane, right_lane = video.get_frame()
            direction = get_desired_direction(left_lane, right_lane, 352, 288)
            x1, y1, x2, y2 = direction[0]

            steeringangle = get_steeringangle(direction)

            if stabilize:
                steeringangle = stabilize_steeringangle(steeringangle,
                                                        last_steeringangle, 5)
                last_steeringangle = steeringangle

            steering(steeringangle, pwm)

            if gen_dataset:
                white = np.ones((288, 352, 1), dtype=np.uint8) * 255
                roi = np.array([[0, 288], [0, 230], [88, 130], [264, 130],
                                [352, 230], [352, 288]])
                stencil = cv2.fillConvexPoly(white, roi, 0)
                stencil2 = np.repeat(stencil[...], 3, -1)
                frame =  cv2.add(frame, stencil2)
                frame_direction = np.copy(frame)

                cv2.line(frame_direction, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(frame_direction, f"steeringangle: {steeringangle}",
                            (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

                time_now = time.asctime().replace(' ', '-').replace(":", "-")
                cv2.imwrite("testing/dataset/img-{}-{:.1f}.png".format(time_now, steeringangle), frame)
                cv2.imwrite("testing/dataset/img-ctrl-{}-{:.1f}.png".format(time_now, steeringangle), frame_direction)
                cv2.imshow("frame-direction", frame_direction)
            else:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            cv2.imshow("frame", frame)
            cv2.imshow("frame w/ lines", frame_lines)
            cv2.imshow("ROI frame", roi_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)
    finally:
        motor_ctrl(0, pwm)
        steering(0, pwm)


def turtle_wrapper(func, max_acc=22, steps=2, stabilize=False, gen_dataset=False):
    ''' wrapper for main() that slows down the car by switching the motor on
        and off in a seperate daemon-thread '''
    def func_wrapper(*args, **kwargs):
        pwm = config_pwm(hz=60)
        motor_thread = threading.Thread(target=go_slow_multistep,
                                        args=(pwm, max_acc, 0.15, steps,),
                                        daemon=True)
        motor_thread.start()
        base_func = func(gen_dataset, stabilize, max_acc)

        # restore base-config
        motor_ctrl(0, pwm)
        steering(0, pwm)

    return func_wrapper


turtle_mode = turtle_wrapper(main, max_acc=30)


if __name__ == "__main__":
    steering_warmup()
    main()
    turtle_mode()
