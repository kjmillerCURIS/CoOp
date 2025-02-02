import os
import sys
import numpy as np
import pickle
import pprint
from scipy.stats import hmean
from tqdm import tqdm

ZEROSHOT_METHODS = ['ZeroshotCLIP', 'ZeroshotCLIP2', 'ZeroshotCLIP2_onetoken']

#NOTE: you can only do this if you have regular domain-splits!
def duplicate_for_zeroshot(agg_results):
    for k in sorted(agg_results.keys()):
        for a in ['seen', 'unseen']:
            if 'unseen_domains_%s_classes'%(a) in agg_results[k]:
                agg_results[k]['seen_domains_%s_classes'%(a)] = agg_results[k]['unseen_domains_%s_classes'%(a)]

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
        for eval_type in tqdm(['seen_domains_seen_classes', 'seen_domains_unseen_classes', 'unseen_domains_seen_classes', 'unseen_domains_unseen_classes']):
            if is_zeroshot and ('unseen_domains' not in eval_type):
                continue

            run_accs = []
            missing_files = []
            all_fewshot_seeds = [0,1,2]
            if is_zeroshot:
                all_fewshot_seeds = [0]

            for fewshot_seed in all_fewshot_seeds:
                seed = 0
                for domain_split_index in range(6):
                    output_dir = os.path.join(experiment_base_dir, 'class_split_%s/fewshot_seed%d/seed%d/domain_split%d/test/%s'%(class_split_type, fewshot_seed, seed, domain_split_index, eval_type))
                    results_filename = os.path.join(output_dir, 'results.pkl')
                    if not os.path.exists(results_filename):
                        missing_files.append(results_filename)
                        continue

                    print(results_filename)
                    with open(results_filename, 'rb') as f:
                        results = pickle.load(f)

                    run_accs.append(results['pergroup_accuracy'])

            if len(missing_files) > 0:
                print('The following files are missing, therefore we cannot compute aggregate accuracy for this setting: %s'%(str(missing_files)))
                continue

            print(run_accs)
            agg_results[class_split_type][eval_type] = np.mean(run_accs)

        if len(agg_results[class_split_type].keys()) == (2 if is_zeroshot else 4):
            nums = [agg_results[class_split_type][k] for k in sorted(agg_results[class_split_type].keys())]
            assert(len(nums) == (2 if is_zeroshot else 4))
            assert('H' not in agg_results[class_split_type])
            agg_results[class_split_type]['H'] = hmean(nums)

    with open(os.path.join(experiment_base_dir, 'agg_results_multiseed.pkl'), 'wb') as f:
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
