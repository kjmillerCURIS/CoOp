import os
import sys
import numpy as np
import pickle
import torch
from dassl.utils import read_image #use this instead of cv2.imread!!!
from dassl.config import get_cfg_default
from dassl.data.transforms import build_transform


#NOTE: LAION images are augmented in the same way as DomainNet images. Each image only gets one augmentation, but we hope that there are enough images for that not to matter. What matters more is that they have the same distribution as whatever we're trying to create with the DomainNet training data. e.g. include random_flip to promote flip invariance, which otherwise might not get promoted even with an infinite amount of unflipped LAION data.


#for augmenting (and preprocessing) image
def get_transform():

    #make cfg for transform
    cfg = get_cfg_default()
    cfg.INPUT.SIZE = (224, 224)
    cfg.INPUT.INTERPOLATION = 'bicubic'
    cfg.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
    cfg.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
    cfg.INPUT.TRANSFORMS = ["random_resized_crop", "random_flip", "normalize"]
    tfm = build_transform(cfg, is_train=True)
    return tfm

class LAIONAugmentedImageDataset(torch.utils.data.Dataset):

    #laion_data_dir should end in something like 'uniform_subset'
    def __init__(self, laion_data_dir):
        self.laion_data_dir = laion_data_dir
        print('loading image_bases.pkl...')
        with open(os.path.join(self.laion_data_dir, 'image_bases.pkl'), 'rb') as f:
            all_image_bases = pickle.load(f)

        print('done loading image_bases.pkl')
        self.image_bases = []
        self.image_ts = []
        for t, image_base in enumerate(all_image_bases):
            image_path = os.path.join(self.laion_data_dir, 'images', '%05d'%(t // 10000), '%09d.jpg'%(t))
            if not os.path.exists(image_path):
                continue

            self.image_bases.append(image_base)
            self.image_ts.append(t)

        self.tfm = get_transform()

    def __len__(self):
        return len(self.image_bases)

    def __getitem__(self, idx):
        t = self.image_ts[idx]
        image_path = os.path.join(self.laion_data_dir, 'images', '%05d'%(t // 10000), '%09d.jpg'%(t))
        assert(os.path.exists(image_path))
        X = read_image(image_path)
        X = self.tfm(X)
        return {'X' : X, 'idx' : torch.tensor(idx, dtype=torch.long)}
