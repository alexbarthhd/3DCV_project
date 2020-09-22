import time

from driving_functions import config_pwm, steering, motor_ctrl


if __name__ == "__main__":
    pwm = config_pwm(hz=60)

    # max right:
    steering(25, pwm)
    time.sleep(3)

    steering(0, pwm)
