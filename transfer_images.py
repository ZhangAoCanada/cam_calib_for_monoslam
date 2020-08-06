import os
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np

# TARGET_SIZE = (640, 480)
TARGET_SIZE = (720, 540)

def main(image_input_dir, image_output_dir, if_resize=False):
    hm_imgs = len(glob(os.path.join(image_input_dir, "*.jpg")))
    for i in tqdm(range(hm_imgs)):
        image_name = os.path.join(image_input_dir, "%.6d.jpg"%(i))
        if not os.path.exists(image_name):
            continue
        image = cv2.imread(image_name)
        if if_resize:
            image = cv2.resize(image, TARGET_SIZE)
        new_img_name = os.path.join(image_output_dir, "%.d.jpg"%(i))
        cv2.imwrite(new_img_name, image)

if __name__ == "__main__":
    image_input_dir = "/Users/aozhang/Downloads/tmp_data/calibrate"
    image_output_dir = "./calib_images"
    if_resize = True
    main(image_input_dir, image_output_dir, if_resize=if_resize)
