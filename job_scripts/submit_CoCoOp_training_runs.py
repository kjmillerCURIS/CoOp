import os
import sys

DEBUG=False
#JOB_NAME_FILTER = ['CoCoOp_train_ds3_csrandom', 'CoCoOp_train_ds4_csrandom', 'CoCoOp_train_ds5_csrandom', 'CoCoOp_train_ds0_csordered', 'CoCoOp_train_ds1_csordered', 'CoCoOp_train_ds2_csordered', 'CoCoOp_train_ds3_csordered', 'CoCoOp_train_ds4_csordered']
JOB_NAME_FILTER = None

def submit_CoCoOp_training_runs():
    for class_split_type in ['random', 'ordered']:
        for seed in [0]:
            fewshot_seed = seed
            for domain_split_index in range(6):
                output_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOp/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/train'%(class_split_type, fewshot_seed, seed, domain_split_index)
                job_name = 'CoCoOp_train_ds%d_cs%s'%(domain_split_index, class_split_type)
                if JOB_NAME_FILTER is not None:
                    if job_name not in JOB_NAME_FILTER:
                        continue

                my_cmd = 'qsub -N %s -v CLASS_SPLIT_TYPE=%s,SEED=%d,FEWSHOT_SEED=%d,DOMAIN_SPLIT_INDEX=%d,OUTPUT_DIR=%s run_CoCoOp_train_generic.sh'%(job_name,class_split_type,seed,fewshot_seed,domain_split_index,output_dir)
                print('submitting training run: "%s"'%(my_cmd))
                os.system(my_cmd)
                if DEBUG:
                    print('DEBUG MODE: let\'s see how that first run goes...')
                    return

if __name__ == '__main__':
    submit_CoCoOp_training_runs()
