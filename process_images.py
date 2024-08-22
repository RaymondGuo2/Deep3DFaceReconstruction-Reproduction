import argparse
import os
import mtcnn_detector
from util.process_exif import remove_exif

def main(args):
    for f in os.listdir(args.image_folder):
        if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png"):
            file_path = os.path.join(args.image_folder, f)
            remove_exif(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='./test_images')
    args = parser.parse_args()
    main(args)