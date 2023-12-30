import util
import json
import datetime
import os

class mobileNetConfig(object):
    """Neural network training config to hold hyperparemters"""
    def __init__(self):
        self.DEVICE = util.check_gpu()
        self.SEED = 42
        self.LR = 0.0005
        self.BATCH = 4
        self.EPOCHS = 100
        self.NUM_CLASSES = 2
        self.IMG_SIZE = (480, 640)
        
        self.WEIGHT_DECAY = 1e-4
        self.L1_LAMBDA = 0.001 
        
        with open('secrets.json', 'r') as file:
            secrets = json.load(file)
            self.BASE_PATH = secrets['models']['mobileNetV2_SSD']
        file.close()
        
        self.CHECKPOINT = os.path.join(self.BASE_PATH, f'{datetime.datetime.now().strftime("%H%M%S_%f")}.checkpoint')
        self.RELOAD = False  
        self.TRAIN = False
        self.TEST = False
        if not os.path.exists(self.BASE_PATH):
            os.makedirs(self.BASE_PATH)
            