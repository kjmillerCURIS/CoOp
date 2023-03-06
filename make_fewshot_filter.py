import os
import sys
import glob
import pickle
import random
from tqdm import tqdm

NUM_SHOTS_PER_DOMAINCLASS = 3

def make_fewshot_filter(domainnet_dir, seed):
    seed = int(seed)
    random.seed(seed)
    txt_filenames = sorted(glob.glob(os.path.join(domainnet_dir, 'splits', '*_train.txt')))
    domainclass2impaths = {}
    for txt_filename in tqdm(txt_filenames):
        domain = os.path.splitext(os.path.basename(txt_filename))[0].split('_')[0]
        f = open(txt_filename, 'r')
        for line in f:
            ss = line.rstrip('\n').split(' ')
            assert(len(ss) == 2)
            impath, classID = ss
            classID = int(classID)
            k = (domain, classID)
            if k not in domainclass2impaths:
                domainclass2impaths[k] = []

            domainclass2impaths[k].append(impath)

    fewshot_filter = []
    for k in tqdm(sorted(domainclass2impaths.keys())):
        if len(domainclass2impaths[k]) < NUM_SHOTS_PER_DOMAINCLASS:
            print('Warning: group (%s, %d) (e.g. "%s") has less than %d samples - so we have to sample from it WITH replacement'%(k[0], k[1], domainclass2impaths[k][0], NUM_SHOTS_PER_DOMAINCLASS))
            fewshots = [random.choice(domainclass2impaths[k]) for _ in range(NUM_SHOTS_PER_DOMAINCLASS)]
        else:
            fewshots = random.sample(domainclass2impaths[k], NUM_SHOTS_PER_DOMAINCLASS)
        
        fewshot_filter.extend(fewshots)

    fewshot_filter_filename = os.path.join(domainnet_dir, 'fewshot_filter_seed%d.pkl'%(seed))
    with open(fewshot_filter_filename, 'wb') as f:
        pickle.dump(set(fewshot_filter), f)

def usage():
    print('Usage: python make_fewshot_filter.py <domainnet_dir> <seed>')

if __name__ == '__main__':
    make_fewshot_filter(*(sys.argv[1:]))
