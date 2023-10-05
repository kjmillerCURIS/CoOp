import os
import sys
import glob
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from laion_imgemb_caption_pair_dataset import LAIONImgembCaptionPairDataset

C_LIST = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
GAMMA_LIST = [1/50000, 1/5000, 3/5000, 1/500, 3/500, 1/50, 1/5]
KERNEL = 'rbf'
N_JOBS = 3

BASE_DIR = '../vislang-domain-exploration-data/CoCoOpExperiments'
IMAGE_DIR = os.path.join(BASE_DIR, 'EDA/TryOutOCR/images')
LAION_DATA_DIR = os.path.join(BASE_DIR, 'laion_data/uniform_subset')
MODEL_FILENAME = os.path.join(BASE_DIR, 'EDA/TryOutOCR/CLIP_OCR_model.pkl')

#X, y should be features and label (1 for text, 0 for no-text)
#return model
def fit_model(X, y):
    pipe = Pipeline([('scaling', StandardScaler()), ('classify', SVC(kernel=KERNEL))])
    param_grid = {'classify__C' : C_LIST, 'classify__gamma' : GAMMA_LIST}
    grid = GridSearchCV(pipe, n_jobs=N_JOBS, param_grid=param_grid, verbose=3)
    grid.fit(X, y)
    return grid

#return X, y which can be passed directly to fit_model()
def load_data(image_dir, laion_data_dir):
    pos_images = sorted(glob.glob(os.path.join(image_dir, 'pos', '*.*')))
    neg_images = sorted(glob.glob(os.path.join(image_dir, 'neg', '*.*')))
    dataset = LAIONImgembCaptionPairDataset(laion_data_dir)
    d = {image_base : p[0] for image_base, p in zip(dataset.image_bases, dataset.pairs)}
    pos_embs = [d[os.path.basename(k)] for k in pos_images]
    neg_embs = [d[os.path.basename(k)] for k in neg_images]
    X_pos = np.array(pos_embs)
    X_neg = np.array(neg_embs)
    X_pos = X_pos / np.sqrt(np.sum(np.square(X_pos), axis=1, keepdims=True))
    X_neg = X_neg / np.sqrt(np.sum(np.square(X_neg), axis=1, keepdims=True))
    X = np.vstack((X_pos, X_neg))
    y = np.zeros(X.shape[0])
    y[:X_pos.shape[0]] = 1
    return X, y

def try_clip_features_to_detect_text(image_dir, laion_data_dir, model_filename):
    X, y = load_data(image_dir, laion_data_dir)
    model = fit_model(X, y)
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

    print(model.cv_results_)

if __name__ == '__main__':
    try_clip_features_to_detect_text(IMAGE_DIR, LAION_DATA_DIR, MODEL_FILENAME)
