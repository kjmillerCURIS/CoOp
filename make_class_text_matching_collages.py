import os
import sys
import cv2
cv2.setNumThreads(1)
import glob
import numpy as np
import pickle
import random
from tqdm import tqdm

#image stuff
NUM_ROWS = 10000000
COLLAGE_WIDTH = 3200
IMAGE_HEIGHT = 224
BUFFER_THICKNESS = 20

#caption stuff
FONT_SIZE = 10
FONT_THICKNESS = 12
TEXT_BAR_HEIGHT = 150
TEXT_BAR_BUFFER_PROP = 0.1
TEXT_BAR_WIDTH_PER_CHAR = 150
MIN_TEXT_BAR_WIDTH_CHARS = 50
TEXT_TRIM_BUFFER = 100
TEXT_RESIZE_FACTOR = 5

def make_collage_one_image_dir(image_dir):
    images = sorted(glob.glob(os.path.join(image_dir, '*', '*.jpg')))
    rows = [[]]
    cur_x = 0
    for image in images:
        numI = cv2.imread(image)
        if numI.shape[0] != IMAGE_HEIGHT:
            w = int(round(numI.shape[1] / numI.shape[0] * IMAGE_HEIGHT))
            numI = cv2.resize(numI, (w, IMAGE_HEIGHT))

        if numI.shape[1] >= COLLAGE_WIDTH:
            print('!!! had to throw away a too-wide image !!!')
            continue

        if cur_x + numI.shape[1] > COLLAGE_WIDTH:
            rows[-1].append(255 * np.ones((IMAGE_HEIGHT, COLLAGE_WIDTH - cur_x, 3), dtype='uint8'))
            cur_x = 0
            if len(rows) >= NUM_ROWS:
                break
            else:
                rows.append([])

        rows[-1].append(numI)
        cur_x += numI.shape[1]
        if cur_x + BUFFER_THICKNESS > COLLAGE_WIDTH:
            rows[-1].append(255 * np.ones((IMAGE_HEIGHT, COLLAGE_WIDTH - cur_x, 3), dtype='uint8'))
            cur_x = 0
            if len(rows) >= NUM_ROWS:
                break
            else:
                rows.append([])

        cur_x += BUFFER_THICKNESS
        rows[-1].append(255 * np.ones((IMAGE_HEIGHT, BUFFER_THICKNESS, 3), dtype='uint8'))

    while len(rows[-1]) == 0:
        rows.pop()

    last_width = sum([x.shape[1] for x in rows[-1]])
    if last_width < COLLAGE_WIDTH:
        rows[-1].append(255 * np.ones((IMAGE_HEIGHT, COLLAGE_WIDTH - last_width, 3), dtype='uint8'))

    stuffs = []
    for row in rows:
        stuffs.append(np.hstack(row))
        stuffs.append(255 * np.ones((BUFFER_THICKNESS, COLLAGE_WIDTH, 3), dtype='uint8'))

    numIcollage = np.vstack(stuffs)
    return numIcollage

def make_class_text_matching_collages(experiment_dir):
    collage_dir = os.path.join(experiment_dir, 'collages')
    os.makedirs(collage_dir, exist_ok=True)
    class_dirs = sorted(glob.glob(os.path.join(experiment_dir, 'images_by_class', '*')))
    for class_dir in tqdm(class_dirs):
        class_name = os.path.basename(class_dir)
        image_dir = os.path.join(class_dir, 'images')
        numIcollage = make_collage_one_image_dir(image_dir)
        cv2.imwrite(os.path.join(collage_dir, class_name + '-collage.jpg'), numIcollage)

def usage():
    print('Usage: python make_class_text_matching_collages.py <experiment_dir>')

if __name__ == '__main__':
    make_class_text_matching_collages(*(sys.argv[1:]))
