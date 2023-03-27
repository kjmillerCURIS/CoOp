import os
import sys
import glob
import pickle
import random

RANDOM_SEED = 0
NUM_CLASSES = 20

DOMAINNET_DIR = '../vislang-domain-exploration-data/DatasetsForCoCoOp/domainnet'
EXPERIMENT_DIR = '../vislang-domain-exploration-data/CoCoOpExperiments/EDA/laion_class_text_matching'

def get_class_name(path):
    s = os.path.basename(path)
    ss = s.replace('-', ' ').replace('_', ' ').split()
    if len(ss) == 1:
        return ss[0]
    else:
        return ss

def pick_random_classes():
    random.seed(RANDOM_SEED)

    all_class_names = [get_class_name(path) for path in sorted(glob.glob(os.path.join(DOMAINNET_DIR, 'real', '*')))]
    class_names = random.sample(all_class_names, NUM_CLASSES)
    print(class_names)

    with open(os.path.join(EXPERIMENT_DIR, 'class_names.pkl'), 'wb') as f:
        pickle.dump(class_names, f)

if __name__ == '__main__':
    pick_random_classes()
