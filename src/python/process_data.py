import util

if __name__ == "__main__":
    image_directories = ['path/to/dir1', 'path/to/dir2', 'path/to/dir3']
    annotations_path = 'path/to/annotation_train.odgt'
    preprocessor = util.Preprocessor(image_directories, annotations_path)
    preprocessor.preprocess_images()