import os
import sys
from friendly_submitter import FriendlySubmitter

DEBUG=False
JOB_NAME_FILTER=None
INTERVAL=2
BATCH_SIZE=14

def submit_CoCoOpEnsemblingLAIONized_gramlen_training_runs():
    my_friend = FriendlySubmitter(INTERVAL, batch_size=BATCH_SIZE)
    for class_split_type in ['random']:
        for fewshot_seed in [0,1,2]:
            seed = 0 #CoCoOp treats "0" as nondeterministic
            for domain_split_index in range(6):
                for gramlen in [1, 3, 5]:
                    output_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOpEnsemblingLAIONized_gramlen%d/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/train'%(gramlen, class_split_type, fewshot_seed, seed, domain_split_index)
                    job_name = 'CoCoOpEnsemblingLAIONized_gramlen%d_train_fs%d_ds%d_cs%s'%(gramlen, fewshot_seed, domain_split_index, class_split_type)
                    if JOB_NAME_FILTER is not None:
                        if job_name not in JOB_NAME_FILTER:
                            continue

                    my_cmd = 'qsub -N %s -v GRAMLEN=%d,CLASS_SPLIT_TYPE=%s,SEED=%d,FEWSHOT_SEED=%d,DOMAIN_SPLIT_INDEX=%d,OUTPUT_DIR=%s run_CoCoOpEnsemblingLAIONized_gramlen_train_generic.sh'%(job_name,gramlen,class_split_type,seed,fewshot_seed,domain_split_index,output_dir)
                    my_friend.add(my_cmd)
                    if DEBUG:
                        print('DEBUG MODE: let\'s see how that first run goes...')
                        my_friend.run()
                        return

    my_friend.run()

if __name__ == '__main__':
    submit_CoCoOpEnsemblingLAIONized_gramlen_training_runs()
