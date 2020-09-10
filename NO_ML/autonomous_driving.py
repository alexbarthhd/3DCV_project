from driving_functions import steering, motor_ctrl


if "__name__" == "__main__":
    # max right:
    steering(25)

    # max left:
    steering(-25)

    # "medium" right:
    steering(12.5)

    # "medium" left:
    steering(-12.5)
