import os
import sys
import clip
import glob
import numpy as np
import pickle
import torch
from tqdm import tqdm

CLIP_MODEL_TYPE = 'ViT-B/16'
BATCH_SIZE = 256

def get_input_shard_filenames(laion_base_dir, start_index, stride):
    input_shard_filenames = sorted(glob.glob(os.path.join(laion_base_dir, 'image_level_info_dict-*.pkl')))
    return input_shard_filenames[start_index::stride]

def get_output_shard_filename(input_shard_filename, experiment_dir):
    output_shard_filename = os.path.join(experiment_dir, 'laion_text_embedding_dict-' + os.path.splitext(os.path.basename(input_shard_filename))[0].split('-')[-1] + '.pkl')
    return output_shard_filename

def embed(captions, clip_model):
    texts = clip.tokenize(captions, truncate=True).to('cuda')
    with torch.no_grad():
        feats = clip_model.encode_text(texts)
        feats = feats / feats.norm(dim=1, keepdim=True)
        feats = feats.cpu().numpy()

    return feats

#output_shard will have following keys:
#-"embeddings" which goes to a single matrix
#-"info" which goes to a list of (image_base, caption, url) tuples
def process_shard(input_shard, clip_model):
    batch = []
    embeddings = []
    info = []
    for k in tqdm(sorted(input_shard.keys())):
        if input_shard[k]['laion_type'] != 'laion2B-en':
            continue

        info.append((k, input_shard[k]['caption'], input_shard[k]['url']))
        batch.append(input_shard[k]['caption'])
        if len(batch) >= BATCH_SIZE:
            feats = embed(batch, clip_model)
            embeddings.append(feats)
            batch = []

    if len(batch) > 0:
        feats = embed(batch, clip_model)
        embeddings.append(feats)

    if len(info) == 0:
        return None

    return {'info' : info, 'embeddings' : np.vstack(embeddings)}

def embed_all_laion_captions(experiment_dir, laion_base_dir, start_index, stride):
    os.makedirs(experiment_dir, exist_ok=True)
    start_index = int(start_index)
    stride = int(stride)

    clip_model, _ = clip.load(CLIP_MODEL_TYPE, device='cuda')
    input_shard_filenames = get_input_shard_filenames(laion_base_dir, start_index, stride)
    for input_shard_filename in input_shard_filenames:
        with open(input_shard_filename, 'rb') as f:
            input_shard = pickle.load(f)

        output_shard = process_shard(input_shard, clip_model)
        if output_shard is None:
            print('"%s" contained no english'%(input_shard_filename))
            continue

        output_shard_filename = get_output_shard_filename(input_shard_filename, experiment_dir)
        with open(output_shard_filename, 'wb') as f:
            pickle.dump(output_shard, f)

def usage():
    print('Usage: python embed_all_laion_captions.py <experiment_dir> <laion_base_dir> <start_index> <stride>')

if __name__ == '__main__':
    embed_all_laion_captions(*(sys.argv[1:]))
