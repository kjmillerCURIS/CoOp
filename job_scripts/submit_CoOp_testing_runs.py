import os
import sys

SKIP_IF_ALREADY_RESULTS = True
DEBUG=False
JOB_NAME_FILTER = None

def submit_CoOp_testing_runs():
    for class_split_type in ['random', 'ordered']:
        for seed in [0]:
            fewshot_seed = seed
            for domain_split_index in range(6):
                model_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoOp/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/train'%(class_split_type, fewshot_seed, seed, domain_split_index)
                for eval_type in ['seen_domains_seen_classes', 'seen_domains_unseen_classes', 'unseen_domains_seen_classes', 'unseen_domains_unseen_classes']:
                    output_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoOp/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/test/%s'%(class_split_type, fewshot_seed, seed, domain_split_index, eval_type)
                    job_name = 'CoOp_test_ds%d_cs%s_et%s'%(domain_split_index, class_split_type, eval_type)
                    if JOB_NAME_FILTER is not None:
                        if job_name not in JOB_NAME_FILTER:
                            continue

                    if SKIP_IF_ALREADY_RESULTS and os.path.exists(os.path.join(output_dir, 'results.pkl')):
                        continue

                    my_cmd = 'qsub -N %s -v CLASS_SPLIT_TYPE=%s,SEED=%d,FEWSHOT_SEED=%d,DOMAIN_SPLIT_INDEX=%d,MODEL_DIR=%s,OUTPUT_DIR=%s,EVAL_TYPE=%s run_CoOp_test_generic.sh'%(job_name,class_split_type,seed,fewshot_seed,domain_split_index,model_dir,output_dir,eval_type)
                    print('submitting testing run: "%s"'%(my_cmd))
                    os.system(my_cmd)
                    if DEBUG:
                        print('DEBUG MODE: let\'s see how that first run goes...')
                        return

if __name__ == '__main__':
    submit_CoOp_testing_runs()
