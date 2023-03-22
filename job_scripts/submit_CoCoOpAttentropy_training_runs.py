import os
import sys

DEBUG=False
JOB_NAME_FILTER = None

def submit_CoCoOpAttentropy_training_runs():
    for class_split_type in ['random']:
        for seed in [0]:
            fewshot_seed = seed
            for domain_split_index in range(6):
#                for attentropy_lambda in ['0_1', '0_5', '1_0']:
#                for attentropy_lambda in ['0_05', '0_2']:
#                for attentropy_lambda in ['0_0125', '0_025', '0_075']:
                for attentropy_lambda in ['0_0375']:
                    output_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOpAttentropy_lambda%s/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/train'%(attentropy_lambda, class_split_type, fewshot_seed, seed, domain_split_index)
                    job_name = 'CoCoOpAttentropy_lambda%s_train_ds%d_cs%s'%(attentropy_lambda, domain_split_index, class_split_type)
                    if JOB_NAME_FILTER is not None:
                        if job_name not in JOB_NAME_FILTER:
                            continue

                    my_cmd = 'qsub -N %s -v ATTENTROPY_LAMBDA=%s,CLASS_SPLIT_TYPE=%s,SEED=%d,FEWSHOT_SEED=%d,DOMAIN_SPLIT_INDEX=%d,OUTPUT_DIR=%s run_CoCoOpAttentropy_train_generic.sh'%(job_name,attentropy_lambda,class_split_type,seed,fewshot_seed,domain_split_index,output_dir)
                    print('submitting training run: "%s"'%(my_cmd))
                    os.system(my_cmd)
                    if DEBUG:
                        print('DEBUG MODE: let\'s see how that first run goes...')
                        return

if __name__ == '__main__':
    submit_CoCoOpAttentropy_training_runs()
