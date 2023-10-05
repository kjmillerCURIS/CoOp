import os
import sys
import random
import shutil
from tqdm import tqdm
from laion_augmented_image_dataset import LAIONAugmentedImageDataset

NUM_IMAGES = 700
RANDOM_SEED = 1337
BASE_DIR = '../vislang-domain-exploration-data/CoCoOpExperiments'
LAION_DATA_DIR = os.path.join(BASE_DIR, 'laion_data/uniform_subset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'EDA/TryOutOCR/images/all2')
OTHER_DIRS = [os.path.join(BASE_DIR, 'EDA/TryOutOCR/images/all'), os.path.join(BASE_DIR, 'EDA/TryOutOCR/images/holdout')]

def grab_laion_random_images(laion_data_dir, num_images, random_seed, output_dir, other_dirs):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(random_seed)
    dataset = LAIONAugmentedImageDataset(laion_data_dir)
    indices = random.sample(range(len(dataset)), num_images)
    for idx in tqdm(indices):
        t = dataset.image_ts[idx]
        image_path = os.path.join(laion_data_dir, 'images', '%05d'%(t // 10000), '%09d.jpg'%(t))
        assert(os.path.exists(image_path))
        image_base = dataset.image_bases[idx]
        if any([os.path.exists(os.path.join(other_dir, image_base)) for other_dir in OTHER_DIRS]):
            continue

        shutil.copy(image_path, os.path.join(output_dir, image_base))

if __name__ == '__main__':
    grab_laion_random_images(LAION_DATA_DIR, NUM_IMAGES, RANDOM_SEED, OUTPUT_DIR, OTHER_DIRS)
