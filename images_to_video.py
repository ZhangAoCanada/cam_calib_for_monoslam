import os
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np

# TARGET_SIZE = (640, 480)
TARGET_SIZE = (720, 540)

def main(image_input_dir, video_output_dir, skip_step, if_resize=False):
    hm_imgs = len(glob(os.path.join(image_input_dir, "*.jpg")))
    sample_image_name = glob(os.path.join(image_input_dir, "*.jpg"))[0]
    sample_image = cv2.imread(sample_image_name)
    if if_resize:
        sample_image = cv2.resize(sample_image, TARGET_SIZE)
    h, w, ch = sample_image.shape
    v = cv2.VideoWriter(video_output_dir, cv2.VideoWriter_fourcc(*'DIVX'), 15, (w, h))
    for i in tqdm(range(hm_imgs)):
        img_name = os.path.join(image_input_dir, "%.6d.jpg"%(i)) 
        if not os.path.exists(img_name):
            continue
        img = cv2.imread(img_name)
        if if_resize:
            img = cv2.resize(img, TARGET_SIZE)
        if i % skip_step == 0:
            v.write(img)
    v.release()

if __name__ == "__main__":
    # image_input_dir = "/Users/aozhang/Downloads/tmp_data/seq"
    image_input_dir = "/Users/aozhang/Downloads/tmp_data/seq2"
    video_output_name = "./result/seq2.avi"
    skip_step = 2
    if_resize = True
    main(image_input_dir, video_output_name, skip_step, if_resize=if_resize)
