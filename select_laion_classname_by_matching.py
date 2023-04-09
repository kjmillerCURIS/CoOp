import os
import sys
import clip
import numpy as np
import pickle
import torch
from tqdm import tqdm
from laion_imgemb_caption_pair_dataset import LAIONImgembCaptionPairDataset
from trainers.cocoop import easy_load_ViTB16_whole
from trainers.efficient_text_encoder import EfficientTextEncoder, EFFICIENT_CONTEXT_LENGTH

MAX_GRAMLEN = 7
BATCH_SIZE = 512
NUM_WORKERS = 2

#tokenized_captions shape should be (N, 77)
#returns tensor with shape (N, 77-f(gramlen), gramlen) where f() is something I'll figure out soon I promise...
#output will NOT have any SOS tokens, but it will have EOS tokens in at least the last row of the "stack"
#it will read "down-is-forwards"
def make_caption_stack(tokenized_captions, gramlen):
    assert(gramlen >= 1)
    if gramlen == 1:
        return tokenized_captions[:,1:].unsqueeze(2)
    else:
        base = tokenized_captions[:,1:]
        stuffs = []
        for g in range(gramlen):
            stuffs.append(base[:,g:g+base.shape[1]-gramlen+1])

        caption_stack = torch.stack(stuffs, dim=2)
        return caption_stack

#takes in output of make_caption_stack()
#returns tokenized_ngrams, indices
#tokenized_ngrams will have the full 77 tokens (someone else can cut it down), and indices will index into the original batch
def unpack_caption_stack(caption_stack):
    gramlen = caption_stack.shape[-1]

    #figure out how many ngrams we're gonna produce
    ngram_count_list = caption_stack[:,:,-1].argmax(dim=1)
    N_out = torch.sum(ngram_count_list).item()

    #make a placeholder
    tokenized_ngrams = torch.tile(clip.tokenize(' '.join(['X' for i in range(gramlen)])).cuda(), (N_out, 1))
    indices = []
    cur_t = 0
    for index, (one_stack, ngram_count) in enumerate(zip(caption_stack, ngram_count_list)):
        indices.extend([index for i in range(ngram_count.item())])
        tokenized_ngrams[cur_t:cur_t+ngram_count.item(), 1:1+gramlen] = one_stack[:ngram_count.item(),:]
        cur_t += ngram_count.item()

    indices = torch.tensor(indices, dtype=torch.long)
    return tokenized_ngrams, indices

def embed_ngrams(tokenized_ngrams, token_embedding, text_encoder):
    ngram_tokens = token_embedding(tokenized_ngrams)
    return text_encoder(ngram_tokens.type(text_encoder.dtype), tokenized_ngrams)

#batch will have keys 'imgemb' and 'caption'
#we'll return a list of dictionarie
#each dictionary will have keys that are the lengths of ngrams, and values (tokenized_ngram, cossim)
#where tokenized_ngram is the efficient length (subsequent datasets can pad it to the full 77 tokens if they want)
#we assume that batch has no empty captions
#the dataset that uses this dictionary can take in a "gramlen" parameter and return the best-scoring ngram whose length is at most gramlen. It's not liike the dictionary itself has to be torch-collatable.
@torch.no_grad()
def process_batch(batch, token_embedding, text_encoder):
    tokenized_captions = batch['caption'].cuda()
    image_features = batch['imgemb'].cuda()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    out = [{gramlen : (None, float('-inf')) for gramlen in range(1, MAX_GRAMLEN + 1)} for i in range(tokenized_captions.shape[0])]
    for gramlen in range(1, MAX_GRAMLEN + 1):
        caption_stack = make_caption_stack(tokenized_captions, gramlen)
        tokenized_ngrams, indices = unpack_caption_stack(caption_stack)
        assert(tokenized_ngrams.argmax(dim=1).max().item() < EFFICIENT_CONTEXT_LENGTH)
        tokenized_ngrams = tokenized_ngrams[:,:EFFICIENT_CONTEXT_LENGTH]
        ngram_features = embed_ngrams(tokenized_ngrams, token_embedding, text_encoder)
        ngram_features = ngram_features / ngram_features.norm(dim=1, keepdim=True)
        image_features_to_compare = image_features[indices, :]
        cossims = torch.sum(ngram_features * image_features_to_compare, dim=1)
        for index, cossim, tokenized_ngram in zip(indices, cossims, tokenized_ngrams):
            if cossim.item() > out[index.item()][gramlen][1]:
                out[index.item()][gramlen] = (tokenized_ngram.cpu().numpy(), cossim.item()) #yes, it's a bit wasteful to call .cpu().numpy() that many times...

    return out

def select_laion_classname_by_matching(laion_data_dir):
    clip_model = easy_load_ViTB16_whole()
    text_encoder = EfficientTextEncoder(clip_model)
    token_embedding = clip_model.token_embedding
    dataset = LAIONImgembCaptionPairDataset(laion_data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    out_dict = {}
    for batch in tqdm(dataloader):
        out = process_batch(batch, token_embedding, text_encoder)
        for outlet, idx in zip(out, batch['idx']):
            out_dict[dataset.image_bases[idx]] = outlet

    with open(os.path.join(laion_data_dir, 'caption_classname_dict.pkl'), 'wb') as f:
        pickle.dump(out_dict, f)

def usage():
    print('Usage: python select_laion_classname_by_matching.py <laion_data_dir>')

if __name__ == '__main__':
    select_laion_classname_by_matching(*(sys.argv[1:]))
