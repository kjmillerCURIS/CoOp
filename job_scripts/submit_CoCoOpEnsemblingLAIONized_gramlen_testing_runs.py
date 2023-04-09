import os
import sys
from friendly_submitter import FriendlySubmitter

SKIP_IF_ALREADY_RESULTS = False
DEBUG=False
JOB_NAME_FILTER = None
NEGATIVE_JOB_NAME_FILTER = None
INTERVAL = 0.75
BATCH_SIZE = 15

def submit_CoCoOpEnsemblingLAIONized_gramlen_testing_runs():
    my_friend = FriendlySubmitter(INTERVAL, batch_size=BATCH_SIZE)
    for class_split_type in ['random']:
        for fewshot_seed in [0,1,2]:
            seed = 0 #CoCoOp treats "0" as nondeterministic
            for domain_split_index in range(6):
                for gramlen in [1, 3, 5]:
                    model_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOpEnsemblingLAIONized_gramlen%d/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/train'%(gramlen, class_split_type, fewshot_seed, seed, domain_split_index)
                    for eval_type in ['seen_domains_seen_classes', 'seen_domains_unseen_classes', 'unseen_domains_seen_classes', 'unseen_domains_unseen_classes']:
                        output_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOpEnsemblingLAIONized_gramlen%d/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/test/%s'%(gramlen, class_split_type, fewshot_seed, seed, domain_split_index, eval_type)
                        job_name = 'CoCoOpEnsemblingLAIONized_gramlen%d_test_fs%d_ds%d_cs%s_et%s'%(gramlen, fewshot_seed, domain_split_index, class_split_type, eval_type) 
                        if JOB_NAME_FILTER is not None:
                            if job_name not in JOB_NAME_FILTER:
                                continue

                        if NEGATIVE_JOB_NAME_FILTER is not None:
                            if job_name in NEGATIVE_JOB_NAME_FILTER:
                                continue

                        if SKIP_IF_ALREADY_RESULTS and os.path.exists(os.path.join('..', output_dir, 'results.pkl')):
                            continue

                        script_name = 'run_CoCoOpEnsemblingLAIONized_gramlen_test_generic.sh'
                        my_cmd = 'qsub -N %s -v GRAMLEN=%d,CLASS_SPLIT_TYPE=%s,SEED=%d,FEWSHOT_SEED=%d,DOMAIN_SPLIT_INDEX=%d,MODEL_DIR=%s,OUTPUT_DIR=%s,EVAL_TYPE=%s %s'%(job_name,gramlen,class_split_type,seed,fewshot_seed,domain_split_index,model_dir,output_dir,eval_type,script_name)
                        my_friend.add(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            my_friend.run()
                            return

    my_friend.run()

if __name__ == '__main__':
    submit_CoCoOpEnsemblingLAIONized_gramlen_testing_runs()
