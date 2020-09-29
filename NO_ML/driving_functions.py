import Adafruit_PCA9685
import time


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
    ''' converts accelersation in % into pwm-signal '''
    if -100 <= acceleration <= 100:
        # forwards:
        if acceleration > 0:
            pulse_length = 1.4 * acceleration + 370

        # backwards:
        elif acceleration < 0:
            pulse_length = 0.6 * acceleration + 370

        # stop:
        elif acceleration == 0:
            pulse_length = 370

        pwm.set_pwm(0, 0, int(pulse_length))
    else:
        print("acceleration out of range")


def go_slow(pwm, max_acc, sleep_time):
    ''' pwm inception '''
    while(1):
        time.sleep(sleep_time)
        motor_ctrl(0, pwm)
        time.sleep(sleep_time)
        motor_ctrl(max_acc, pwm)


def go_slow_multistep(pwm, max_acc, sleep_time, steps):
    while(1):
        pwm_step = max_acc // steps
        pwm_val = max_acc

        for i in range(steps - 1):
            motor_ctrl(pwm_val, pwm)
            pwm_val -= pwm_step
            time.sleep(sleep_time)

        motor_ctrl(0, pwm)

        for i in range(steps - 1):
            motor_ctrl(pwm_val, pwm)
            pwm_val += pwm_step
            time.sleep(sleep_time)

        motor_ctrl(max_acc, pwm)
