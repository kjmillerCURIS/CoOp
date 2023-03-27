import os
import sys
from friendly_submitter import FriendlySubmitter

DEBUG=True
JOB_NAME_FILTER = None
INTERVAL = 0.75
BATCH_SIZE = 12

def submit_CoCoOpMultimodal_training_runs():
    my_friend = FriendlySubmitter(INTERVAL, batch_size=BATCH_SIZE)
#    for class_split_type in ['random', 'ordered']:
    for class_split_type in ['random']:
        for fewshot_seed in [0,1,2]:
            seed = 0 #CoCoOp uses "0" as sentinel for nondeterministic
            for domain_split_index in range(6):
                output_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOpMultimodal/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/train'%(class_split_type, fewshot_seed, seed, domain_split_index)
                job_name = 'CoCoOpMultimodal_train_fs%d_ds%d_cs%s'%(fewshot_seed, domain_split_index, class_split_type)
                if JOB_NAME_FILTER is not None:
                    if job_name not in JOB_NAME_FILTER:
                        continue

                my_cmd = 'qsub -N %s -v CLASS_SPLIT_TYPE=%s,SEED=%d,FEWSHOT_SEED=%d,DOMAIN_SPLIT_INDEX=%d,OUTPUT_DIR=%s run_CoCoOpMultimodal_train_generic.sh'%(job_name,class_split_type,seed,fewshot_seed,domain_split_index,output_dir)
#                print('submitting training run: "%s"'%(my_cmd))
#                os.system(my_cmd)
                my_friend.add(my_cmd)
                if DEBUG:
                    print('DEBUG MODE: let\'s see how that first run goes...')
                    my_friend.run()
                    return

    my_friend.run()

if __name__ == '__main__':
    submit_CoCoOpMultimodal_training_runs()
