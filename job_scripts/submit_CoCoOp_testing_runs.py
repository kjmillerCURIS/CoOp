import os
import sys

SKIP_IF_ALREADY_RESULTS = False
DEBUG=True
JOB_NAME_FILTER = None
RECORD_ATTENTROPY = False

def submit_CoCoOp_testing_runs():
#    for class_split_type in ['random', 'ordered']:
    for class_split_type in ['random']:
#        for fewshot_seed in [0]:
        for fewshot_seed in [1,2]:
            seed = 0 #CoCoOp uses "0" as sentinel for nondeterministic
            for domain_split_index in range(6):
                model_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOp/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/train'%(class_split_type, fewshot_seed, seed, domain_split_index)
                for eval_type in ['seen_domains_seen_classes', 'seen_domains_unseen_classes', 'unseen_domains_seen_classes', 'unseen_domains_unseen_classes']:
                    if RECORD_ATTENTROPY and not (class_split_type == 'random' and eval_type == 'unseen_domains_unseen_classes'):
                        continue

                    output_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOp/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/test/%s'%(class_split_type, fewshot_seed, seed, domain_split_index, eval_type)
                    job_name = 'CoCoOp_test_fs%d_ds%d_cs%s_et%s'%(fewshot_seed, domain_split_index, class_split_type, eval_type)
                    if RECORD_ATTENTROPY:
                        job_name = 'CoCoOp_test_attentropy_fs%d_ds%d_cs%s_et%s'%(fewshot_seed, domain_split_index, class_split_type, eval_type)
                    
                    if JOB_NAME_FILTER is not None:
                        if job_name not in JOB_NAME_FILTER:
                            continue

                    if SKIP_IF_ALREADY_RESULTS and os.path.exists(os.path.join('..', output_dir, 'results.pkl')):
                        continue

                    script_name = 'run_CoCoOp_test_generic.sh'
                    if RECORD_ATTENTROPY:
                        script_name = 'run_CoCoOp_test_record_attentropy.sh'

                    my_cmd = 'qsub -N %s -v CLASS_SPLIT_TYPE=%s,SEED=%d,FEWSHOT_SEED=%d,DOMAIN_SPLIT_INDEX=%d,MODEL_DIR=%s,OUTPUT_DIR=%s,EVAL_TYPE=%s %s'%(job_name,class_split_type,seed,fewshot_seed,domain_split_index,model_dir,output_dir,eval_type,script_name)
                    print('submitting testing run: "%s"'%(my_cmd))
                    os.system(my_cmd)
                    if DEBUG:
                        print('DEBUG MODE: let\'s see how that first run goes...')
                        return

if __name__ == '__main__':
    submit_CoCoOp_testing_runs()
