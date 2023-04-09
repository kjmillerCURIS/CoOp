import os
import sys

def submit_embed_all_laion_captions():
    for i in range(10):
        os.system('qsub -N embed_all_laion_captions_%d -v START_INDEX=%d run_embed_all_laion_captions.sh'%(i, i))

if __name__ == '__main__':
    submit_embed_all_laion_captions()
