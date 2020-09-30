import datetime
import json
from pathlib import Path
from PIL import Image

class DataWriter:
    def __init__(self, data_path = 'data'):
        '''
        Initialize tub for new image record pairs.
        '''
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        todays_tubs = list(Path(data_path).glob(f'*{today}*'))
        tub_nr = max({int(str(tub)[-1]) for tub in todays_tubs}) + 1
        self.tub_path = Path(data_path)/f'tub-{today}-{tub_nr}'
        self.tub_path.mkdir()
        self.record_nr = 0

    def save(self, image, steering, throttle):
        '''
        Save an image record pair to the tub.
        '''
        image_path = self.tub_path/f'image_{self.record_nr}.jpg'
        im = Image.fromarray(image)
        im.save(image_path)
        record = {
            'image': str(image_path),
            'steering': steering,
            'throttle': throttle
        }
        record_path = self.tub_path/f'record_{self.record_nr}.json'
        record_path.write_text(json.dumps(record))
        self.record_nr += 1