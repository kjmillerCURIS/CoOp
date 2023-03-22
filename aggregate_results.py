import os
import sys
import numpy as np
import pickle
import pprint
from tqdm import tqdm

ZEROSHOT_METHODS = ['ZeroshotCLIP', 'ZeroshotCLIP2', 'ZeroshotCLIP2_onetoken']

#NOTE: you can only do this if you have regular domain-splits!
def duplicate_for_zeroshot(agg_results):
    for k in sorted(agg_results.keys()):
        for kk in sorted(agg_results[k].keys()):
            for kkk in sorted(agg_results[k][kk].keys()):
                for a in ['seen', 'unseen']:
                    if 'unseen_domains_%s_classes'%(a) in agg_results[k][kk][kkk]:
                        agg_results[k][kk][kkk]['seen_domains_%s_classes'%(a)] = agg_results[k][kk][kkk]['unseen_domains_%s_classes'%(a)]

    return agg_results

#experiment_base_dir should have two subdirs named "train" and "test" (or at least "test")
#this script will write its output to experiment_base_dir/agg_results.pkl
def aggregate_results(experiment_base_dir):
    is_zeroshot = (os.path.basename(experiment_base_dir) in ZEROSHOT_METHODS)
    if is_zeroshot:
        print('Zeroshot method detected! Will only compute "unseen_domains" settings and then copy them over to "seen_domains" settings when displaying (but will only save the "unseen_domains" results')

    agg_results = {}
#    for class_split_type in ['random', 'ordered']:
    for class_split_type in ['random']:
        agg_results[class_split_type] = {}
        for seed in [0]:
            fewshot_seed = seed
            agg_results[class_split_type][fewshot_seed] = {seed : {}}
            for eval_type in tqdm(['seen_domains_seen_classes', 'seen_domains_unseen_classes', 'unseen_domains_seen_classes', 'unseen_domains_unseen_classes']):
                if is_zeroshot and ('unseen_domains' not in eval_type):
                    continue

                domain_split_accs = []
                missing_files = []
                for domain_split_index in range(6):
                    output_dir = os.path.join(experiment_base_dir, 'class_split_%s/fewshot_seed%d/seed%d/domain_split%d/test/%s'%(class_split_type, fewshot_seed, seed, domain_split_index, eval_type))
                    results_filename = os.path.join(output_dir, 'results.pkl')
                    if not os.path.exists(results_filename):
                        missing_files.append(results_filename)
                        continue

                    with open(results_filename, 'rb') as f:
                        results = pickle.load(f)

                    domain_split_accs.append(results['pergroup_accuracy'])

                if len(missing_files) > 0:
                    print('The following files are missing, therefore we cannot compute aggregate accuracy for this setting: %s'%(str(missing_files)))
                    continue

                agg_results[class_split_type][fewshot_seed][seed][eval_type] = np.mean(domain_split_accs)

    with open(os.path.join(experiment_base_dir, 'agg_results.pkl'), 'wb') as f:
        pickle.dump(agg_results, f)
    
    if is_zeroshot:
        agg_results = duplicate_for_zeroshot(agg_results)

    print('')
    pp = pprint.PrettyPrinter(indent=4, compact=True)
    pp.pprint(agg_results)
    print('')
    print('')

def usage():
    print('Usage: python aggregate_results.py <experiment_base_dir>')

if __name__ == '__main__':
    aggregate_results(*(sys.argv[1:]))
