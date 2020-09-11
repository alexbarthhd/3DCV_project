import Adafruit_PCA9685


def steering(angle):
    ''' converts steeringangle in degrees into pwm-signal '''
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(60)

    if -25 <= angle <= 25:
        # left:
        if angle < 0:
            pulse_length = (-5/22) * angle + 380

        # right:
        elif angle > 0:
            pulse_length = (-18/5) * angle + 380

        # center:
        elif angle == 0:
            pulse_length = 380

        pwm.set_pwm(1, 0, int(pulse_length))
    else:
        print("angle out of range")


def motor_ctrl(acceleration):
    ''' converts acceleration in % into pwm-signal '''
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(60)

    if -100 <= acceleration <= 100:
        # forwards:
        if acceleration > 0:
            pass

        # backwards:
        elif acceleration < 0:
            pass

        # stop:
        elif acceleration == 0:
            pass
    else:
        print("acceleration out of range")
