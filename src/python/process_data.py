import util
import os

if __name__ == "__main__":
    image_directories = ['./src/python/CrowdHuman_train01/Images/', './src/python/CrowdHuman_train02/Images/', './src/python/CrowdHuman_train03/Images/']
    annotations_path = './src/python/annotation_train.odgt'
    preprocessor = util.Preprocessor(image_directories, annotations_path, './src/python/')
    preprocessor.preprocess_images()