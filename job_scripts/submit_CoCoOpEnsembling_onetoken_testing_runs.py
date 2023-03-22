import os
import sys

SKIP_IF_ALREADY_RESULTS = True
DEBUG=False
JOB_NAME_FILTER = None
NEGATIVE_JOB_NAME_FILTER = ['CoCoOpEnsembling_random_separate_onetoken_test_ds0_csrandom_etseen_domains_seen_classes', 'CoCoOpEnsembling_random_separate_onetoken_test_ds0_csrandom_etseen_domains_unseen_classes']

def submit_CoCoOpEnsembling_onetoken_testing_runs():
    for class_split_type in ['random']:
        for seed in [0]:
            fewshot_seed = seed
            for domain_split_index in range(6):
                for random_or_manual in ['random', 'manual']:
#                    for separate_or_together in ['separate', 'together']:
                    for separate_or_together in ['separate']:
                        for onetoken_type in ['onetoken', 'onetokennoperiod']:
                            model_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOpEnsembling_%s_%s_%s/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/train'%(random_or_manual, separate_or_together, onetoken_type, class_split_type, fewshot_seed, seed, domain_split_index)
                            for eval_type in ['seen_domains_seen_classes', 'seen_domains_unseen_classes', 'unseen_domains_seen_classes', 'unseen_domains_unseen_classes']:
                                output_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOpEnsembling_%s_%s_%s/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/test/%s'%(random_or_manual, separate_or_together, onetoken_type, class_split_type, fewshot_seed, seed, domain_split_index, eval_type)
                                job_name = 'CoCoOpEnsembling_%s_%s_%s_test_ds%d_cs%s_et%s'%(random_or_manual, separate_or_together, onetoken_type, domain_split_index, class_split_type, eval_type) 
                                if JOB_NAME_FILTER is not None:
                                    if job_name not in JOB_NAME_FILTER:
                                        continue

                                if NEGATIVE_JOB_NAME_FILTER is not None:
                                    if job_name in NEGATIVE_JOB_NAME_FILTER:
                                        continue

                                if SKIP_IF_ALREADY_RESULTS and os.path.exists(os.path.join('..', output_dir, 'results.pkl')):
                                    continue

                                script_name = 'run_CoCoOpEnsembling_onetoken_test_generic.sh'
                                my_cmd = 'qsub -N %s -v RANDOM_OR_MANUAL=%s,SEPARATE_OR_TOGETHER=%s,ONETOKEN_TYPE=%s,CLASS_SPLIT_TYPE=%s,SEED=%d,FEWSHOT_SEED=%d,DOMAIN_SPLIT_INDEX=%d,MODEL_DIR=%s,OUTPUT_DIR=%s,EVAL_TYPE=%s %s'%(job_name,random_or_manual,separate_or_together,onetoken_type,class_split_type,seed,fewshot_seed,domain_split_index,model_dir,output_dir,eval_type,script_name)
                                print('submitting testing run: "%s"'%(my_cmd))
                                os.system(my_cmd)
                                if DEBUG:
                                    print('DEBUG MODE: let\'s see how that first run goes...')
                                    return

if __name__ == '__main__':
    submit_CoCoOpEnsembling_onetoken_testing_runs()
