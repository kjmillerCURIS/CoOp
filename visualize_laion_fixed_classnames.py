import os
import sys
import cv2
import glob
import pickle
import random
from tqdm import tqdm
import clip
from caption_image import caption_image

IMAGE_BLOWUP = 2
FONT_SIZE_MULTIPLIER = 14 / 600
MIN_FONT_SIZE = 14

#caption_image(numI, caption_lines, font_size=16, width_ratio=0.9, height_buffer=0.1):

def detokenize(toks, decoder):
    return ''.join([decoder[tok].replace('</w>', ' ') for tok in toks if tok not in [0, 49406, 49407]]).strip()

def make_caption_lines(d, decoder, laioncap):
    caption_lines = ['LAION caption: "%s"'%(laioncap)]
    for gramlen in sorted(d.keys()):
        if d[gramlen][0] is None:
            continue

        for_human = detokenize(d[gramlen][0], decoder)
        caption_lines.append('Best %d-gram (cossim=%.5f): "%s"'%(gramlen, d[gramlen][1], for_human))

    return caption_lines

def load_image(image_path):
    numI = cv2.imread(image_path)
    if IMAGE_BLOWUP != 1:
        numI = cv2.resize(numI, None, fx=IMAGE_BLOWUP, fy=IMAGE_BLOWUP)

    return numI

def load_impath_dict(laion_data_dir):
    print('loading image_bases.pkl...')
    with open(os.path.join(laion_data_dir, 'image_bases.pkl'), 'rb') as f:
        all_image_bases = pickle.load(f)

    print('done loading image_bases.pkl')
    impath_dict = {}
    for t, image_base in enumerate(all_image_bases):
        image_path = os.path.join(laion_data_dir, 'images', '%05d'%(t // 10000), '%09d.jpg'%(t))
        if not os.path.exists(image_path):
            continue

        impath_dict[image_base] = image_path

    return impath_dict

def make_visualizations(fixed_classname_dict, impath_dict, laioncap_dict, num_images, vis_dir):
    os.makedirs(vis_dir, exist_ok=True)
    random.seed(0)
    image_bases = random.sample(sorted(fixed_classname_dict.keys()), num_images)
    decoder = clip.simple_tokenizer.SimpleTokenizer().decoder
    for image_base in tqdm(image_bases):
        image_path = impath_dict[image_base]
        numI = load_image(image_path)
        assert(numI is not None)
        d = fixed_classname_dict[image_base]
        caption_lines = make_caption_lines(d, decoder, laioncap_dict[image_base])
        numIvis = caption_image(numI, caption_lines, font_size=max(int(round(FONT_SIZE_MULTIPLIER * numI.shape[1])), MIN_FONT_SIZE))
        vis_filename = os.path.join(vis_dir, image_base)
        cv2.imwrite(vis_filename, numIvis)

def visualize_laion_fixed_classnames(laion_data_dir, num_images, vis_dir):
    num_images = int(num_images)

    with open(os.path.join(laion_data_dir, 'caption_classname_dict.pkl'), 'rb') as f:
        fixed_classname_dict = pickle.load(f)

    with open(os.path.join(laion_data_dir, 'image_base_to_caption.pkl'), 'rb') as f:
        laioncap_dict = pickle.load(f)

    impath_dict = load_impath_dict(laion_data_dir)
    make_visualizations(fixed_classname_dict, impath_dict, laioncap_dict, num_images, vis_dir)

def usage():
    print('Usage: python visualize_laion_fixed_classnames.py <laion_data_dir> <num_images> <vis_dir>')

if __name__ == '__main__':
    visualize_laion_fixed_classnames(*(sys.argv[1:]))
