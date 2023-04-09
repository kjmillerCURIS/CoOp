import os
import sys
import cv2
import glob
import pickle
from tqdm import tqdm
from caption_image import caption_image

IMAGE_BLOWUP = 2
FONT_SIZE_MULTIPLIER = 14 / 600
MIN_FONT_SIZE = 14

#caption_image(numI, caption_lines, font_size=16, width_ratio=0.9, height_buffer=0.1):

def get_out_filenames(experiment_dir):
    return sorted(glob.glob(os.path.join(experiment_dir, '*-nearest_laion_texts.pkl')))

def format_result_dir(result_dir):
    s = result_dir
    s = s.replace('vislang-domain-exploration-data/CoCoOpExperiments/baselines', '')
    s = s.replace('.', '')
    s = s.strip('/')
    s = s.replace('/', '-')
    return s

#out = {'info_list' : info_list, 'stat_list' : stat_list, 'max_cossims' : max_cossims, 'outfo_list' : outfo_list}
#group it so that out_grouped[(result_dir, image_path)][stat_type][stuff] = thing
#stuff could be 'classname', 'clf_cossim', 'clf_prob', 'prompt_str', 'laion_caption', 'laion_max_cossim'
def group_out(out):
    out_grouped = {}
    for info, stat, max_cossim, outfo in zip(out['info_list'], out['stat_list'], out['max_cossims'], out['outfo_list']):
        k = (info[0], info[1])
        if k not in out_grouped:
            out_grouped[k] = {}

        stat_type = info[2]
        assert(stat_type not in out_grouped[k])
        v = {'classname' : stat['name'], 'clf_cossim' : stat['cossim'], 'clf_prob' : stat['prob'], 'prompt_str' : stat['prompt_str'], 'laion_caption' : outfo[1], 'laion_max_cossim' : max_cossim}
        out_grouped[k][stat_type] = v

    return out_grouped

def make_caption_lines(d):
    caption_lines = []
    caption_lines.append('nearest-prompt-tokens="%s"'%(d['gt']['prompt_str'].replace(d['gt']['classname'].replace('_', ' '), '[CLASSNAME]')))
    caption_lines.append('')
    for stat_type in ['gt', 'argmax', 'argmed']:
        caption_lines.append('%s="%s", clf_cossim=%.3f, clf_prob=%.3f'%(stat_type, d[stat_type]['classname'], d[stat_type]['clf_cossim'], d[stat_type]['clf_prob']))
        caption_lines.append('nearest-laion-caption (cossim=%.3f): "%s"'%(d[stat_type]['laion_max_cossim'], d[stat_type]['laion_caption']))
        if stat_type != 'argmed':
            caption_lines.append('')

    return caption_lines

def load_image(image_path):
    numI = cv2.imread(image_path)
    if IMAGE_BLOWUP != 1:
        numI = cv2.resize(numI, None, fx=IMAGE_BLOWUP, fy=IMAGE_BLOWUP)

    return numI

def process_out(out, experiment_dir):
    out_grouped = group_out(out)
    for k in sorted(out_grouped.keys()):
        result_dir, image_path = k
        vis_dir = os.path.join(experiment_dir, 'laion_nearest_text_vis', format_result_dir(result_dir))
        os.makedirs(vis_dir, exist_ok=True)
        numI = load_image(image_path)
        caption_lines = make_caption_lines(out_grouped[k])
        numIvis = caption_image(numI, caption_lines, font_size=max(int(round(FONT_SIZE_MULTIPLIER * numI.shape[1])), MIN_FONT_SIZE))
        vis_filename = os.path.join(vis_dir, os.path.basename(image_path))
        cv2.imwrite(vis_filename, numIvis)

def visualize_nearest_laion_texts(experiment_dir):
    out_filenames = get_out_filenames(experiment_dir)
    for out_filename in tqdm(out_filenames):
        with open(out_filename, 'rb') as f:
            out = pickle.load(f)

        process_out(out, experiment_dir)

def usage():
    print('Usage: python visualize_nearest_laion_texts.py <experiment_dir>')

if __name__ == '__main__':
    visualize_nearest_laion_texts(*(sys.argv[1:]))
