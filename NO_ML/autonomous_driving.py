import time

from driving_functions import config_pwm, steering, motor_ctrl


if __name__ == "__main__":
    pwm = config_pwm(hz=60)
    # max right:
    steering(25, pwm)
    time.sleep(3)

    # max left:
    steering(-25, pwm)
    time.sleep(3)
    steering(-25)
    time.sleep(3)

    # "medium" right:
    steering(25, pwm)
    time.sleep(3)
    steering(12.5, pwm)
    time.sleep(3)

    # "medium" left:
    steering(-12.5, pwm)
    time.sleep(3)
    steering(-12.5, pwm)
