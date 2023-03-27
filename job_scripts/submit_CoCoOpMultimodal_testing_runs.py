import os
import sys
from friendly_submitter import FriendlySubmitter

SKIP_IF_ALREADY_RESULTS = True
DEBUG=False
JOB_NAME_FILTER = None
INTERVAL = 0.75
BATCH_SIZE = 12

def submit_CoCoOpMultimodal_testing_runs():
    my_friend = FriendlySubmitter(INTERVAL, batch_size=BATCH_SIZE)
#    for class_split_type in ['random', 'ordered']:
    for class_split_type in ['random']:
        for fewshot_seed in [0,1,2]:
            seed = 0 #CoCoOp uses "0" as sentinel for nondeterministic
            for domain_split_index in range(6):
                model_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOpMultimodal/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/train'%(class_split_type, fewshot_seed, seed, domain_split_index)
                for eval_type in ['seen_domains_seen_classes', 'seen_domains_unseen_classes', 'unseen_domains_seen_classes', 'unseen_domains_unseen_classes']:
                    output_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOpMultimodal/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/test/%s'%(class_split_type, fewshot_seed, seed, domain_split_index, eval_type)
                    job_name = 'CoCoOpMultimodal_test_fs%d_ds%d_cs%s_et%s'%(fewshot_seed, domain_split_index, class_split_type, eval_type) 
                    if JOB_NAME_FILTER is not None:
                        if job_name not in JOB_NAME_FILTER:
                            continue

                    if SKIP_IF_ALREADY_RESULTS and os.path.exists(os.path.join('..', output_dir, 'results.pkl')):
                        continue

                    script_name = 'run_CoCoOpMultimodal_test_generic.sh'
                    my_cmd = 'qsub -N %s -v CLASS_SPLIT_TYPE=%s,SEED=%d,FEWSHOT_SEED=%d,DOMAIN_SPLIT_INDEX=%d,MODEL_DIR=%s,OUTPUT_DIR=%s,EVAL_TYPE=%s %s'%(job_name,class_split_type,seed,fewshot_seed,domain_split_index,model_dir,output_dir,eval_type,script_name)
#                    print('submitting testing run: "%s"'%(my_cmd))
#                    os.system(my_cmd)
                    my_friend.add(my_cmd)
                    if DEBUG:
                        print('DEBUG MODE: let\'s see how that first run goes...')
                        my_friend.run()
                        return

    my_friend.run()

if __name__ == '__main__':
    submit_CoCoOpMultimodal_testing_runs()
