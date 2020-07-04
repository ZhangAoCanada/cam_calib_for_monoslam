import cv2
import numpy as np
from tqdm import tqdm
import os
from glob import glob

def checkoutDir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        for f in glob(os.path.join(dir_name, "*.jpg")):
            os.remove(f)

def main(video_path, image_dir):
    checkoutDir(image_dir)
    v = cv2.VideoCapture(video_path)
    if not v.isOpened():
        raise ValueError("The video does not exist")

    count = 0
    while(v.isOpened()):
        ret, frame = v.read()
        if ret == True:
            cv2.imwrite(os.path.join(image_dir, "%d.jpg"%(count)), frame)
            print("current image number: \t", count)
            count += 1

if __name__ == "__main__":
    video_path = "/Users/aozhang/Desktop/calib.mov"
    image_dir = "./calib_images"
    main(video_path, image_dir)
