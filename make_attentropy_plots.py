import os
import sys
import numpy as np
import pickle
import pprint
from scipy.stats import linregress
from tqdm import tqdm
from matplotlib import pyplot as plt

BASELINE_DIR = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines'
COCOOP_DIR = os.path.join(BASELINE_DIR, 'CoCoOp')
ZEROSHOT_DIR = os.path.join(BASELINE_DIR, 'ZeroshotCLIP')
PLOT_DIR = 'attentropy_plots'
CLASS_SPLIT_TYPE = 'random'
FEWSHOT_SEED = 0
SEED = 0
EVAL_TYPE = 'unseen_domains_unseen_classes'

MIN_X_WIDTH = 0.5

def make_attentropy_plots():
    os.makedirs(PLOT_DIR, exist_ok=True)
    for attentropy_type in ['avg_attentropy', 'correct_attentropy', 'pred_attentropy']:
        for xy_type in ['CoCoOp', 'ZeroshotCLIP', 'diff']:
            x = []
            y = []
            impaths = set([])
            for domain_split_index in tqdm(range(6)):
                suffix = 'class_split_%s/fewshot_seed%d/seed%d/domain_split%d/test/%s'%(CLASS_SPLIT_TYPE, FEWSHOT_SEED, SEED, domain_split_index, EVAL_TYPE)
                c_pkl = os.path.join(COCOOP_DIR, suffix, 'results.pkl')
                z_pkl = os.path.join(ZEROSHOT_DIR, suffix, 'results.pkl')
                with open(c_pkl, 'rb') as f:
                    c_d = pickle.load(f)

                with open(z_pkl, 'rb') as f:
                    z_d = pickle.load(f)

                for impath in sorted(c_d['details'].keys()):
                    assert(impath not in impaths)
                    impaths.add(impath)
                    if xy_type == 'CoCoOp':
                        x.append(c_d['details'][impath][attentropy_type])
                        y.append(c_d['details'][impath]['test_loss'])
                    elif xy_type == 'ZeroshotCLIP':
                        x.append(z_d['details'][impath][attentropy_type])
                        y.append(z_d['details'][impath]['test_loss'])
                    elif xy_type == 'diff':
                        x.append(c_d['details'][impath][attentropy_type] - z_d['details'][impath][attentropy_type])
                        y.append(c_d['details'][impath]['test_loss'] - z_d['details'][impath]['test_loss'])
                    else:
                        assert(False)

            res = linregress(x,y,alternative='less')

            plt.clf()
            plt.scatter(x,y,marker='.',s=10)
            my_xlim = plt.xlim()
            if my_xlim[1] - my_xlim[0] < MIN_X_WIDTH:
                c = 0.5 * (my_xlim[0] + my_xlim[1])
                my_xlim = (c - 0.5 * MIN_X_WIDTH, c + 0.5 * MIN_X_WIDTH)

            plt.plot(my_xlim, res.slope * np.array(my_xlim) + res.intercept, linestyle='--', color='k')
            xy_type_str = xy_type
            if xy_type == 'diff':
                xy_type_str = 'CoCoOp - ZeroshotCLIP'

            plt.xlabel(xy_type_str + ' ' + attentropy_type)
            plt.ylabel(xy_type_str + ' test_loss')
            plt.title('y = %.3fx + %.3f, R2=%.3f, p=%.3f'%(res.slope, res.intercept, res.rvalue ** 2, res.pvalue))
            plt.xlim(my_xlim)
            plt.savefig(os.path.join(PLOT_DIR, xy_type + '-' + attentropy_type + '.png'))
            plt.clf()

if __name__ == '__main__':
    make_attentropy_plots()
