import os
import sys
import numpy as np
from tqdm import tqdm

#figure out how long most LAION captions are so we can choose a max length that trades off between compute and completeness
#will save count vector of number of tokens in caption, including both <SOS> and <EOS> (but not including the context that we would insert)

def get_laion_caption_stats(laion_data_dir):
    assert(False) #KEVIN

def usage():
    print('Usage: python get_laion_caption_stats.py <laion_data_dir>')

if __name__ == '__main__':
    get_laion_caption_stats(*(sys.argv[1:]))
