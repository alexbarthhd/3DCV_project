import Adafruit_PCA9685


def steering(angle):
    ''' converts steeringangle into pwm-singnal '''
    pwm = Adafruit_PCA9685.PCA9685()

    if -25 <= angle <= 25:
        # left:
        if angle < 0:
            pulse_length = (-5/22) * angle + (950/11)

        # right:
        elif angle > 0:
            pulse_length = (-5/18) * angle + (950/9)

        # center:
        elif angle == 0:
            pulse_length = 380

        pwm.set_pwm(1, 0, pulse_length)
    else:
        print("angle out of range")
