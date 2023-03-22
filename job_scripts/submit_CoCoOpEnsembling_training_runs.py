import os
import sys

DEBUG=False
JOB_NAME_FILTER = ['CoCoOpEnsembling_random_separate_train_ds0_csrandom', 'CoCoOpEnsembling_manual_separate_train_ds4_csrandom', 'CoCoOpEnsembling_manual_together_train_ds4_csrandom', 'CoCoOpEnsembling_random_separate_train_ds5_csrandom']

def submit_CoCoOpEnsembling_training_runs():
    for class_split_type in ['random']:
        for seed in [0]:
            fewshot_seed = seed
            for domain_split_index in range(6):
                for random_or_manual in ['random', 'manual']:
                    for separate_or_together in ['separate', 'together']:
                        output_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOpEnsembling_%s_%s/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/train'%(random_or_manual, separate_or_together, class_split_type, fewshot_seed, seed, domain_split_index)
                        job_name = 'CoCoOpEnsembling_%s_%s_train_ds%d_cs%s'%(random_or_manual, separate_or_together, domain_split_index, class_split_type)
                        if JOB_NAME_FILTER is not None:
                            if job_name not in JOB_NAME_FILTER:
                                continue

                        my_cmd = 'qsub -N %s -v CLASS_SPLIT_TYPE=%s,SEED=%d,FEWSHOT_SEED=%d,DOMAIN_SPLIT_INDEX=%d,OUTPUT_DIR=%s,RANDOM_OR_MANUAL=%s,SEPARATE_OR_TOGETHER=%s run_CoCoOpEnsembling_train_generic.sh'%(job_name,class_split_type,seed,fewshot_seed,domain_split_index,output_dir,random_or_manual,separate_or_together)
                        print('submitting training run: "%s"'%(my_cmd))
                        os.system(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            return

if __name__ == '__main__':
    submit_CoCoOpEnsembling_training_runs()
