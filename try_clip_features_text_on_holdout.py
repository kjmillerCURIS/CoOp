import os
import sys
import glob
import numpy as np
import pickle
import random
import shutil
from tqdm import tqdm
from laion_augmented_image_dataset import LAIONAugmentedImageDataset
from laion_imgemb_caption_pair_dataset import LAIONImgembCaptionPairDataset

BASE_DIR = '../vislang-domain-exploration-data/CoCoOpExperiments'
TRAIN_DIR = os.path.join(BASE_DIR, 'EDA/TryOutOCR/images/all')
HOLDOUT_DIR = os.path.join(BASE_DIR, 'EDA/TryOutOCR/images/holdout')
LAION_DATA_DIR = os.path.join(BASE_DIR, 'laion_data/uniform_subset')
MODEL_FILENAME = os.path.join(BASE_DIR, 'EDA/TryOutOCR/CLIP_OCR_model.pkl')
OUT_DIR = os.path.join(BASE_DIR, 'EDA/TryOutOCR/CLIPFeatures')

RANDOM_SEED = 42
NUM_IMAGES = 1000

def load_data(holdout_dir, laion_data_dir):
    images = sorted(glob.glob(os.path.join(holdout_dir, '*.*')))
    dataset = LAIONImgembCaptionPairDataset(laion_data_dir)
    d = {image_base : p[0] for image_base, p in zip(dataset.image_bases, dataset.pairs)}
    embs = [d[os.path.basename(k)] for k in images]
    X = np.array(embs)
    X = X / np.sqrt(np.sum(np.square(X), axis=1, keepdims=True))
    return X, images

def grab_holdout_images(laion_data_dir, train_dir, num_images, holdout_dir):
    os.makedirs(holdout_dir, exist_ok=True)
    random.seed(RANDOM_SEED)
    train_image_bases = [os.path.basename(image) for image in sorted(glob.glob(os.path.join(train_dir, '*.*')))]
    dataset = LAIONAugmentedImageDataset(laion_data_dir)
    indices = random.sample(range(len(dataset)), num_images)
    for idx in tqdm(indices):
        t = dataset.image_ts[idx]
        image_path = os.path.join(laion_data_dir, 'images', '%05d'%(t // 10000), '%09d.jpg'%(t))
        assert(os.path.exists(image_path))
        image_base = dataset.image_bases[idx]
        if image_base in train_image_bases:
            continue

        shutil.copy(image_path, os.path.join(holdout_dir, image_base))

def try_clip_features_text_on_holdout(train_dir, num_images, holdout_dir, laion_data_dir, model_filename, out_dir):
    os.makedirs(os.path.join(out_dir, 'holdout_pred_pos'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'holdout_pred_neg'), exist_ok=True)
    num_images = int(num_images)
    grab_holdout_images(laion_data_dir, train_dir, num_images, holdout_dir)
    X, images = load_data(holdout_dir, laion_data_dir)
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)

    preds = model.predict(X)
    for image, pred in zip(images, preds):
        if pred == 0:
            shutil.copy(image, os.path.join(out_dir, 'holdout_pred_neg'))
        elif pred == 1:
            shutil.copy(image, os.path.join(out_dir, 'holdout_pred_pos'))
        else:
            assert(False)

if __name__ == '__main__':
    try_clip_features_text_on_holdout(TRAIN_DIR, NUM_IMAGES, HOLDOUT_DIR, LAION_DATA_DIR, MODEL_FILENAME, OUT_DIR)
