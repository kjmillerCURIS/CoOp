import os
import sys
import numpy as np
import pickle
import torch
from tqdm import tqdm
from laion_augmented_image_dataset import LAIONAugmentedImageDataset
from trainers.cocoop import easy_load_ViTB16

BATCH_SIZE = 512
NUM_WORKERS = 6

def embed_augmented_laion_images(laion_data_dir):
    image_encoder = easy_load_ViTB16()
    dataset = LAIONAugmentedImageDataset(laion_data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    emb_dict = {}
    for batch in tqdm(dataloader):
        with torch.no_grad():
            X = batch['X'].cuda().type(torch.float16)
            embs = image_encoder(X)
            assert(embs.dtype == torch.float16)
            embs = embs.cpu().numpy()

        for idx, emb in zip(batch['idx'], embs):
            emb_dict[dataset.image_bases[idx.item()]] = emb

    with open(os.path.join(laion_data_dir, 'augmented_image_embedding_dict.pkl'), 'wb') as f:
        pickle.dump(emb_dict, f)

def usage():
    print('Usage: python embed_augmented_laion_images.py <laion_data_dir>')

if __name__ == '__main__':
    embed_augmented_laion_images(*(sys.argv[1:]))
