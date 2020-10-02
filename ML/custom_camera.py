from donkeycar.parts.dgym import DonkeyGymEnv
import numpy as np
import cv2

class CustomDonkeyGymEnv(DonkeyGymEnv):
    def __init__(self, sim_path, host="127.0.0.1", port=9091, headless=0, env_name="donkey-generated-track-v0", sync="asynchronous", conf={}, delay=0):
        super().__init__(sim_path, host, port, headless, env_name, sync, conf, delay)


    def transform_frame(self, frame, fs):
        for f in fs:
            frame = f(frame)
        return frame

    def update(self):        
        while self.running:
            self.frame, _, _, self.info = self.env.step(self.action)