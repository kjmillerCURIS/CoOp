import os
import sys
import glob
from tqdm import tqdm
from img2dataset import download

IMAGE_SIZE = 224
PROCESS_COUNT = 12
ENCODE_FORMAT = 'jpg'
ENCODE_QUALITY = 95

def download_class_text_matching_images(experiment_dir):
    class_dirs = sorted(glob.glob(os.path.join(experiment_dir, 'images_by_class', '*')))
    for class_dir in tqdm(class_dirs):
        url_filename = os.path.join(class_dir, 'urls.txt')
        output_dir = os.path.join(class_dir, 'images')
        os.makedirs(output_dir, exist_ok=True)
        download(image_size=IMAGE_SIZE, processes_count=PROCESS_COUNT, url_list=url_filename, output_folder=output_dir, output_format='files', input_format='txt', resize_mode='keep_ratio', encode_format=ENCODE_FORMAT, encode_quality=ENCODE_QUALITY)

def usage():
    print('Usage: python download_class_text_matching_images.py <experiment_dir>')

if __name__ == '__main__':
    download_class_text_matching_images(*(sys.argv[1:]))
