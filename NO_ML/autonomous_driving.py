import time

from driving_functions import steering, motor_ctrl


if "__name__" == "__main__":
    # max right:
    steering(25)
    time.sleep(3)

    # max left:
    steering(-25)
    time.sleep(3)

    # "medium" right:
    steering(12.5)
    time.sleep(3)

    # "medium" left:
    steering(-12.5)
