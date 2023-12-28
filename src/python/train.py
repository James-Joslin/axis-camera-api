import json
import os
import re
import glob
import util
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open('./secrets.json', 'r') as file:
        secrets = json.load(file)
    file.close()
    annotations_dir = secrets['imageAnnotations']
    # util.string_to_json(annotations_dir)
        
    images_dir = secrets['imagePixelData']
    for i, anno_set in enumerate(glob.glob("./src/python/*.json")):
        with open(anno_set, 'r') as file:
            annos = json.load(file)
            for ele in annos:
                image = cv2.imread(os.path.join(images_dir, ele['image']))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                ground_truth = ele['boxes']
                print(ground_truth)
                plt.imshow(image)
        file.close()