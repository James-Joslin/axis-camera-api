import util
from network_config import SqueezeNetConfig
import os
import json

if __name__ == "__main__":
    config = SqueezeNetConfig()
    with open('./secrets.json', 'r') as file:
        secrets = json.load(file)
    file.close()

    # train
    util.Preprocessor(
        [
            './src/python/CrowdHuman_train01/Images/', 
            './src/python/CrowdHuman_train02/Images/', 
            './src/python/CrowdHuman_train03/Images/'
        ], 
        './src/python/annotation_train.odgt', 
        './src/python/train_images/', 
        './src/python/train_annotations/',
        config.IMG_SIZE
        ).preprocess_images()
    
    # val
    util.Preprocessor(
        [
            './src/python/CrowdHuman_val/Images/'
        ], 
        './src/python/annotation_val.odgt', 
        './src/python/val_images/', 
        './src/python/val_annotations/',
        config.IMG_SIZE
    ).preprocess_images()