import Adafruit_PCA9685


def config_pwm(hz):
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(hz)

    return pwm


def steering(angle, pwm):
    ''' converts steeringangle in degrees into pwm-signal '''
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


def motor_ctrl(acceleration, pwm):
    ''' converts acceleration in % into pwm-signal '''
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
