import os
import sys
import glob
import clip
from matplotlib import pyplot as plt

DOMAINNET_DIR = '../vislang-domain-exploration-data/DatasetsForCoCoOp/domainnet'
PLOT_FILENAME = 'classname_length_histogram.png'

def plot_classname_length_histogram():
    class_dirs = sorted(glob.glob(os.path.join(DOMAINNET_DIR, 'real', '*')))
    vs = []
    for class_dir in class_dirs:
        classname = os.path.basename(class_dir).replace('_', ' ')
        vs.append(clip.tokenize([classname])[0,:].argmax().item() - 1)

    plt.clf()
    plt.hist(vs)
    plt.savefig(PLOT_FILENAME)
    plt.clf()

if __name__ == '__main__':
    plot_classname_length_histogram()
