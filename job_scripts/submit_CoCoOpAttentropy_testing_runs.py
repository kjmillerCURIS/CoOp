import os
import sys

SKIP_IF_ALREADY_RESULTS = True
DEBUG=False
JOB_NAME_FILTER = None

def submit_CoCoOpAttentropy_testing_runs():
    for class_split_type in ['random']:
        for seed in [0]:
            fewshot_seed = seed
            for domain_split_index in range(6):
#                for attentropy_lambda in ['0_1', '0_5', '1_0']:
#                for attentropy_lambda in ['0_05', '0_2']:
#                for attentropy_lambda in ['0_0125', '0_025', '0_075']:
                for attentropy_lambda in ['0_0375']:
                    model_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOpAttentropy_lambda%s/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/train'%(attentropy_lambda, class_split_type, fewshot_seed, seed, domain_split_index)
                    for eval_type in ['seen_domains_seen_classes', 'seen_domains_unseen_classes', 'unseen_domains_seen_classes', 'unseen_domains_unseen_classes']:

                        output_dir = '../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOpAttentropy_lambda%s/class_split_%s/fewshot_seed%d/seed%d/domain_split%d/test/%s'%(attentropy_lambda, class_split_type, fewshot_seed, seed, domain_split_index, eval_type)
                        job_name = 'CoCoOpAttentropy_lambda%s_test_ds%d_cs%s_et%s'%(attentropy_lambda, domain_split_index, class_split_type, eval_type)
                    
                        if JOB_NAME_FILTER is not None:
                            if job_name not in JOB_NAME_FILTER:
                                continue

                        if SKIP_IF_ALREADY_RESULTS and os.path.exists(os.path.join('..', output_dir, 'results.pkl')):
                            continue

                        script_name = 'run_CoCoOpAttentropy_test_generic.sh'
                        my_cmd = 'qsub -N %s -v ATTENTROPY_LAMBDA=%s,CLASS_SPLIT_TYPE=%s,SEED=%d,FEWSHOT_SEED=%d,DOMAIN_SPLIT_INDEX=%d,MODEL_DIR=%s,OUTPUT_DIR=%s,EVAL_TYPE=%s %s'%(job_name,attentropy_lambda,class_split_type,seed,fewshot_seed,domain_split_index,model_dir,output_dir,eval_type,script_name)
                        print('submitting testing run: "%s"'%(my_cmd))
                        os.system(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            return

if __name__ == '__main__':
    submit_CoCoOpAttentropy_testing_runs()
