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
            pulse_length = -4.4 * angle + 380

        # right:
        elif angle > 0:
            pulse_length = -3.6 * angle + 380

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
            pulse_length = 1.4 * acceleration + 360

        # backwards:
        elif acceleration < 0:
            pulse_length = 0.6 * acceleration + 360

        # stop:
        elif acceleration == 0:
            pulse_length = 360
            
        pwm.set_pwm(0, 0, int(pulse_length))
    else:
        print("acceleration out of range")
