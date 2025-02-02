import os
import sys

DEBUG=False
JOB_NAME_FILTER = None

def submit_CoOp_training_runs():
#    for class_split_type in ['random', 'ordered']:
    for class_split_type in ['random']:
#        for fewshot_seed in [0]:
        for fewshot_seed in [1,2]:
            seed = 0 #CoCoOp treats "0" as nondeterministic
            for domain_split_index in range(6):
                output_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoOp/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/train'%(class_split_type, fewshot_seed, seed, domain_split_index)
                job_name = 'CoOp_train_fs%d_ds%d_cs%s'%(fewshot_seed, domain_split_index, class_split_type)
                if JOB_NAME_FILTER is not None:
                    if job_name not in JOB_NAME_FILTER:
                        continue

                my_cmd = 'qsub -N %s -v CLASS_SPLIT_TYPE=%s,SEED=%d,FEWSHOT_SEED=%d,DOMAIN_SPLIT_INDEX=%d,OUTPUT_DIR=%s run_CoOp_train_generic.sh'%(job_name,class_split_type,seed,fewshot_seed,domain_split_index,output_dir)
                print('submitting training run: "%s"'%(my_cmd))
                os.system(my_cmd)
                if DEBUG:
                    print('DEBUG MODE: let\'s see how that first run goes...')
                    return

if __name__ == '__main__':
    submit_CoOp_training_runs()
