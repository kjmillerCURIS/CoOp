import os
import sys
import glob
import numpy as np
import pickle
import time
from tqdm import tqdm
from write_to_log_file import write_to_log_file

#return info_list, query_mat, stat_list
#info_list should be list of tuples, each one (result_dir, image_path, stat_type)
#query_mat should be shape (N, emb_dim) where N = len(info)
#stat_list should be a list of the stat dicts
def load_query(algo_dir, fewshot_seed, domain_split_index):
    info_list = []
    query_mat = []
    stat_list = []
    query_filenames = sorted(glob.glob(os.path.join(algo_dir, 'class_split_random', 'fewshot_seed%d'%(fewshot_seed), 'seed0', 'domain_split%d'%(domain_split_index), 'test', '*', 'results.pkl')))
    for query_filename in query_filenames:
        result_dir = os.path.dirname(query_filename)
        with open(query_filename, 'rb') as f:
            query_dict = pickle.load(f)

        query_dict = query_dict['examples']
        for image_path in sorted(query_dict.keys()):
            for stat_type in ['gt', 'argmax', 'argmed']:
                info_list.append((result_dir, image_path, stat_type))
                query_mat.append(query_dict[image_path][stat_type]['text_emb'])
                stat_list.append({kk : query_dict[image_path][stat_type][kk] for kk in ['name', 'cossim', 'prob', 'prompt_str']})

    query_mat = np.array(query_mat)
    return info_list, query_mat, stat_list

def get_ref_filenames(experiment_dir):
    return sorted(glob.glob(os.path.join(experiment_dir, 'laion_text_embedding_dict-*.pkl')))

#return ref_mat, ref_list
def load_ref(ref_filename):
    with open(ref_filename, 'rb') as f:
        ref_dict = pickle.load(f)

    return ref_dict['embeddings'], ref_dict['info']

def make_out_filename(experiment_dir, algo_dir, fewshot_seed, domain_split_index):
    return os.path.join(experiment_dir, os.path.basename(algo_dir) + '-fewshot_seed%d'%(fewshot_seed) + '-domain_split%d'%(domain_split_index) + '-nearest_laion_texts.pkl')

#query_mat should be shape (N, emb_dim)
#ref_mat should be shape (M, emb_dim)
#ref_list should be list of tuples of length M
#max_cossims should be vector of length N
#outfo_list should be list of tuples (same type as ref_list) of length N
#return max_cossims, outfo_list
def compare(query_mat, ref_mat, ref_list, max_cossims, outfo_list):
    cossims = query_mat @ ref_mat.T
    indices = np.argmax(cossims, axis=1)
    scores = cossims[np.arange(cossims.shape[0]),indices]
    replace = (scores > max_cossims)
    for i in np.nonzero(replace)[0]:
        max_cossims[i] = scores[i]
        outfo_list[i] = ref_list[indices[i]]

    return max_cossims, outfo_list

#algo dir would have basename like "CoCoOp"
def find_nearest_laion_texts(experiment_dir, algo_dir, fewshot_seed, domain_split_index):
    fewshot_seed = int(fewshot_seed)
    domain_split_index = int(domain_split_index)

    info_list, query_mat, stat_list = load_query(algo_dir, fewshot_seed, domain_split_index)
    ref_filenames = get_ref_filenames(experiment_dir)
    max_cossims = -np.inf * np.ones(query_mat.shape[0])
    outfo_list = [None for i in range(query_mat.shape[0])]
    for ref_filename in tqdm(ref_filenames):
        ref_mat, ref_list = load_ref(ref_filename)
        max_cossims, outfo_list = compare(query_mat, ref_mat, ref_list, max_cossims, outfo_list)
        write_to_log_file(str(time.time()))

    out = {'info_list' : info_list, 'stat_list' : stat_list, 'max_cossims' : max_cossims, 'outfo_list' : outfo_list}
    out_filename = make_out_filename(experiment_dir, algo_dir, fewshot_seed, domain_split_index)
    with open(out_filename, 'wb') as f:
        pickle.dump(out, f)

def usage():
    print('Usage: python find_nearest_laion_texts.py <experiment_dir> <algo_dir> <fewshot_seed> <domain_split_index>')

if __name__ == '__main__':
    find_nearest_laion_texts(*(sys.argv[1:]))
