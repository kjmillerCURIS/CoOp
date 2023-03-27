import os
import sys
import glob
import numpy as np
import pickle
import random
from tqdm import tqdm

RANDOM_SEED = 0
NUM_IMAGES_PER_CLASS = 100
SHORTLIST_PROB = 0.05

def format_class_name(x):
    if isinstance(x, list):
        return '_'.join(x)
    else:
        return x

#will populate shortlists with (image_base, url, caption) tuples
def process_shard(image_shard, text_matching_shard, class_names, shortlists, counts):
    for image_base in sorted(image_shard.keys()):
        match_vec = text_matching_shard[image_base]
        if np.amax(match_vec) == 0:
            continue

        for is_match, class_name in zip(match_vec, class_names):
            if is_match:
                counts[class_name] = counts[class_name] + 1
                r = random.uniform(0,1)
                if r <= SHORTLIST_PROB:
                    url = image_shard[image_base]['url']
                    caption = image_shard[image_base]['caption']
                    shortlists[class_name].append((image_base, url, caption))

    return shortlists, counts

def pick_from_shortlists(shortlists):
    picklists = {}
    for k in sorted(shortlists.keys()):
        if len(shortlists[k]) < NUM_IMAGES_PER_CLASS:
            print('Only have %d images for class "%s"'%(len(shortlists[k]), k))
            picklists[k] = shortlists[k]
        else:
            picklists[k] = random.sample(shortlists[k], NUM_IMAGES_PER_CLASS)

    return picklists

def process_picklists(picklists, experiment_dir):
    for k in sorted(picklists.keys()):
        os.makedirs(os.path.join(experiment_dir, 'images_by_class', k, 'images'), exist_ok=True)
        picks = picklists[k]
        f = open(os.path.join(experiment_dir, 'images_by_class', k, 'urls.txt'), 'w')
        for pick in picks:
            f.write(pick[1] + '\n')

        f.close()
        with open(os.path.join(experiment_dir, 'images_by_class', k, 'picks.pkl'), 'wb') as f:
            pickle.dump(picks, f)

def pick_images_from_text_matching(experiment_dir, laion_base_dir):
    random.seed(RANDOM_SEED)

    with open(os.path.join(experiment_dir, 'class_names.pkl'), 'rb') as f:
        class_names = pickle.load(f)

    class_names = [format_class_name(x) for x in class_names]
    shortlists = {k : [] for k in class_names}
    counts = {k : 0 for k in class_names}

    image_level_info_shard_filenames = sorted(glob.glob(os.path.join(laion_base_dir, 'image_level_info_dict-*.pkl')))
    for image_level_info_shard_filename in tqdm(image_level_info_shard_filenames):
        text_matching_shard_filename = os.path.join(experiment_dir, 'class_text_matching_dict-' + os.path.splitext(os.path.basename(image_level_info_shard_filename))[0].split('-')[-1] + '.pkl')
        with open(image_level_info_shard_filename, 'rb') as f:
            image_shard = pickle.load(f)

        with open(text_matching_shard_filename, 'rb') as f:
            text_matching_shard = pickle.load(f)

        shortlists, counts = process_shard(image_shard, text_matching_shard, class_names, shortlists, counts)

    print('\nCounts:')
    print(str(counts) + '\n')
    picklists = pick_from_shortlists(shortlists)
    process_picklists(picklists, experiment_dir)

def usage():
    print('Usage: python pick_images_from_text_matching.py <experiment_dir> <laion_base_dir>')

if __name__ == '__main__':
    pick_images_from_text_matching(*(sys.argv[1:]))
