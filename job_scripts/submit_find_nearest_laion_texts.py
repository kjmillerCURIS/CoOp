import os
import sys

def submit_find_nearest_laion_texts():
    for i in range(6):
        os.system('qsub -N find_nearest_laion_texts_%d -v DOMAIN_SPLIT_INDEX=%d run_find_nearest_laion_texts.sh'%(i, i))

if __name__ == '__main__':
    submit_find_nearest_laion_texts()
