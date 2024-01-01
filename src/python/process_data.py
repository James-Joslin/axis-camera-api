import util
import os

if __name__ == "__main__":
    image_directories = ['./src/python/CrowdHuman_train01/Images/', './src/python/CrowdHuman_train02/Images/', './src/python/CrowdHuman_train03/Images/']
    preprocessor = util.Preprocessor(image_directories, './src/python/annotation_train.odgt', './src/python/train_images/', './src/python/train_annotations/')
    preprocessor.preprocess_images()
    
    preprocessor = util.Preprocessor('./src/python/CrowdHuman_val/Images/', './src/python/annotation_val.odgt', './src/python/val_images/', './src/python/val_annotations/')
    preprocessor.preprocess_images()