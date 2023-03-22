import os
import sys
import pickle
from aggregate_results import ZEROSHOT_METHODS, duplicate_for_zeroshot

NAME_DICT = {'ZeroshotCLIP2' : 'ZeroshotCLIPEnsembling', 'ZeroshotCLIP2_onetoken' : 'ZeroshotCLIPEnsembling_onetoken'}
EVAL_TYPES = ['seen_domains_seen_classes', 'seen_domains_unseen_classes', 'unseen_domains_seen_classes', 'unseen_domains_unseen_classes']
FEWSHOT_SEED = 0
SEED = 0

def load_results(experiment_dir):
    with open(os.path.join(experiment_dir, 'agg_results.pkl'), 'rb') as f:
        agg_results = pickle.load(f)

    if os.path.basename(experiment_dir) in ZEROSHOT_METHODS:
        agg_results = duplicate_for_zeroshot(agg_results)

    return agg_results

def tablify_results_for_3_15_2023_update(experiment_dirs, table_prefix, comparison_experiment_dir=None):
    table_dir = os.path.dirname(table_prefix)
    os.makedirs(table_dir, exist_ok=True)
    
    experiment_dirs = experiment_dirs.split(',')
    if comparison_experiment_dir is not None:
        comparison_results = load_results(comparison_experiment_dir)

    for class_split_type in ['random']:
        f = open(table_prefix + '-' + class_split_type + '-class_split.csv', 'w')
        f.write(',' + ','.join(EVAL_TYPES) + '\n')
        for experiment_dir in experiment_dirs:
            items = []
            exp_name = os.path.basename(experiment_dir)
            print(exp_name)
            if exp_name in NAME_DICT:
                exp_name = NAME_DICT[exp_name]
            elif 'CoCoOpAttentropy_lambda' in exp_name:
                exp_name = 'CoCoOpAttentropy (lambda=%s)'%(exp_name.split('lambda')[1].replace('_', '.'))

            items.append(exp_name)
            results = load_results(experiment_dir)
            for eval_type in EVAL_TYPES:
                res = results[class_split_type][FEWSHOT_SEED][SEED][eval_type]
                if comparison_experiment_dir is not None:
                    compres = comparison_results[class_split_type][FEWSHOT_SEED][SEED][eval_type]
                    diff = res - compres
                    my_sign = ''
                    if diff < 0.0:
                        my_sign = '-'
                    elif diff > 0.0:
                        my_sign = '+'

                    items.append(my_sign + '%.3f%%'%(diff))
                else:
                    items.append('%.3f%%'%(res))

            f.write(','.join(items) + '\n')

        f.close()

def usage():
    print('Usage: python tablify_results_for_3_15_2023_update.py <experiment_dirs> <table_prefix> [<comparison_experiment_dir>=None]')

if __name__ == '__main__':
    tablify_results_for_3_15_2023_update(*(sys.argv[1:]))
