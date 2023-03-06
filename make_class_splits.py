import os
import sys
import glob
import pickle
import random
from tqdm import tqdm

''' Does both "ordered" and "random" splits '''

SEED = 0

def make_class_splits(domainnet_dir):
    random.seed(SEED)
    
    #first we find the classes
    txt_filenames = sorted(glob.glob(os.path.join(domainnet_dir, 'splits', '*_train.txt')))
    classes = set([])
    for txt_filename in tqdm(txt_filenames):
        f = open(txt_filename, 'r')
        for line in f:
            ss = line.rstrip('\n').split(' ')
            assert(len(ss) == 2)
            _, classID = ss
            classID = int(classID)
            classes.add(classID)

    classes = sorted(classes)
    N_seen = len(classes) // 2

    #"ordered" split
    seen, unseen = classes[:N_seen], classes[N_seen:]
    with open(os.path.join(domainnet_dir, 'class_split_ordered.pkl'), 'wb') as f:
        pickle.dump({'seen' : seen, 'unseen' : unseen}, f)

    #"random" split
    seen = sorted(random.sample(classes, N_seen))
    unseen = [classID for classID in classes if classID not in seen]
    with open(os.path.join(domainnet_dir, 'class_split_random.pkl'), 'wb') as f:
        pickle.dump({'seen' : seen, 'unseen' : unseen}, f)

def usage():
    print('Usage: python make_class_splits.py <domainnet_dir>')

if __name__ == '__main__':
    make_class_splits(*(sys.argv[1:]))
