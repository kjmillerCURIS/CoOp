import os
import sys
import numpy as np
import pickle
from aggregate_results import ZEROSHOT_METHODS, duplicate_for_zeroshot

NAME_DICT = {'ZeroshotCLIP2' : 'ZeroshotCLIPEnsembling', 'ZeroshotCLIP2_onetoken' : 'ZeroshotCLIPEnsembling_onetoken', 'CLIPAdapter' : 'CLIPAdapter (alpha=0.2 (default))', 'CLIPAdapter_alpha0_5' : 'CLIPAdapter (alpha=0.5)', 'CLIPAdapter_alpha0_8' : 'CLIPAdapter (alpha=0.8)'}
EVAL_TYPES = ['seen_domains_seen_classes', 'seen_domains_unseen_classes', 'unseen_domains_seen_classes', 'unseen_domains_unseen_classes', 'H']
FEWSHOT_SEED = 0
SEED = 0

def load_results(experiment_dir):
    with open(os.path.join(experiment_dir, 'agg_results_multiseed.pkl'), 'rb') as f:
        agg_results = pickle.load(f)

    if os.path.basename(experiment_dir) in ZEROSHOT_METHODS:
        agg_results = duplicate_for_zeroshot(agg_results)

    return agg_results

#rows is list of rows, not including column names
#first item in each row is experiment name (as string), other items are accuracies (as float)
def write_table(rows, table_filename):
    f = open(table_filename, 'w')
    f.write(',' + ','.join(EVAL_TYPES) + '\n')
    top2 = np.argsort(-np.array([row[1:] for row in rows]), axis=0)[:2,:]
    for i, row in enumerate(rows):
        items = [row[0]]
        for j, z in enumerate(row[1:]):
            if top2[0,j] == i:
                items.append('FIRST%.3f%%'%(z))
            elif top2[1,j] == i:
                items.append('SECOND%.3f%%'%(z))
            else:
                items.append('%.3f%%'%(z))

        f.write(','.join(items) + '\n')

    f.close()

def tablify_results_for_3_29_2023_update(experiment_dirs, table_prefix):
    table_dir = os.path.dirname(table_prefix)
    os.makedirs(table_dir, exist_ok=True)
    
    if not isinstance(experiment_dirs, list):
        assert(isinstance(experiment_dirs, str))
        experiment_dirs = experiment_dirs.split(',')

    for class_split_type in ['random']:
        table_filename = table_prefix + '-' + class_split_type + '-class_split.csv'
        rows = []
        for experiment_dir in experiment_dirs:
            row = []
            exp_name = os.path.basename(experiment_dir)
            print(exp_name)
            if exp_name in NAME_DICT:
                exp_name = NAME_DICT[exp_name]
            elif 'CoCoOpAttentropy_lambda' in exp_name:
                exp_name = 'CoCoOpAttentropy (lambda=%s)'%(exp_name.split('lambda')[1].replace('_', '.'))

            row.append(exp_name)
            results = load_results(experiment_dir)
            for eval_type in EVAL_TYPES:
                res = results[class_split_type][eval_type]
                row.append(res)

            rows.append(row)

        write_table(rows, table_filename)

if __name__ == '__main__':
    table_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/tables_for_3_29_2023_update'
    exp_base_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines'

    #repeat of baselines
    tablify_results_for_3_29_2023_update([os.path.join(exp_base_dir, exp) for exp in ['ZeroshotCLIP', 'ZeroshotCLIP2', 'CLIPAdapter', 'CLIPAdapter_alpha0_5', 'CLIPAdapter_alpha0_8', 'CoOp', 'CoCoOp']], os.path.join(table_dir, 'baseline_repeats'))
    
    #repeat of ensembling + onetoken
    tablify_results_for_3_29_2023_update([os.path.join(exp_base_dir, exp) for exp in ['ZeroshotCLIP', 'ZeroshotCLIP2', 'CoCoOp', 'CoCoOpEnsembling_manual_separate', 'CoCoOpEnsembling_manual_separate_onetokennoperiod', 'CoCoOpEnsembling_random_separate', 'CoCoOpEnsembling_random_separate_onetokennoperiod', 'ZeroshotCLIP2_onetoken']], os.path.join(table_dir, 'ensembling_and_onetoken_repeats'))

    #repeat of attentropy
    tablify_results_for_3_29_2023_update([os.path.join(exp_base_dir, exp) for exp in ['ZeroshotCLIP', 'ZeroshotCLIP2', 'CoCoOp', 'CoCoOpAttentropy_lambda0_0125', 'CoCoOpAttentropy_lambda0_025', 'CoCoOpAttentropy_lambda0_0375', 'CoCoOpAttentropy_lambda0_05', 'CoCoOpAttentropy_lambda0_075', 'CoCoOpAttentropy_lambda0_1']], os.path.join(table_dir, 'attentropy_repeats'))

    #multimodal with repeats
    tablify_results_for_3_29_2023_update([os.path.join(exp_base_dir, exp) for exp in ['ZeroshotCLIP', 'ZeroshotCLIP2', 'CoCoOp', 'CoCoOpMultimodal']], os.path.join(table_dir, 'multimodal_with_repeats'))
