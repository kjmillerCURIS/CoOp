import os
import sys
import glob
import numpy as np
import pickle
import string
from tqdm import tqdm

def preprocess_caption(caption):
    caption = caption.lower()
    for punc in string.punctuation + '0123456789':
        caption = caption.replace(punc, ' ')

    words = caption.split()
    return words

def is_match_one_target(words, target):
    if isinstance(target, list):
        return ('-' + '-'.join(target) + '-' in '-' + '-'.join(words) + '-')
    else:
        return target in words

#returns dict mapping each image_base to a list of ints, 0 or 1, in same order as class_names
def process_one_shard(image_level_info_shard, class_names):
    text_matching_shard = {}
    for image_base in tqdm(sorted(image_level_info_shard.keys())):
        caption = image_level_info_shard[image_base]['caption']
        words = preprocess_caption(caption)
        out_vec = []
        for class_name in class_names:
            out_vec.append(int(is_match_one_target(words, class_name)))

        text_matching_shard[image_base] = np.array(out_vec)

    return text_matching_shard

def do_class_text_matching(experiment_dir, laion_base_dir, start_index=0, stride=16):
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    laion_base_dir = os.path.abspath(os.path.expanduser(laion_base_dir))
    start_index = int(start_index)
    stride = int(stride)

    with open(os.path.join(experiment_dir, 'class_names.pkl'), 'rb') as f:
        class_names = pickle.load(f)

    image_level_info_shard_filenames = sorted(glob.glob(os.path.join(laion_base_dir, 'image_level_info_dict-*.pkl')))
    for image_level_info_shard_filename in tqdm(image_level_info_shard_filenames[start_index::stride]):
        text_matching_shard_filename = os.path.join(experiment_dir, 'class_text_matching_dict-' + os.path.splitext(os.path.basename(image_level_info_shard_filename))[0].split('-')[-1] + '.pkl')
        with open(image_level_info_shard_filename, 'rb') as f:
            image_level_info_shard = pickle.load(f)

        text_matching_shard = process_one_shard(image_level_info_shard, class_names)
        with open(text_matching_shard_filename, 'wb') as f:
            pickle.dump(text_matching_shard, f)

def usage():
    print('Usage: python do_class_text_matching.py <experiment_dir> <laion_base_dir> [<start_index>=0] [<stride>=16]')

if __name__ == '__main__':
    do_class_text_matching(*(sys.argv[1:]))
