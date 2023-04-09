import os
import sys
import clip #clip.tokenize() just uses SimpleTokenizer and then zero-pads and/or truncates as needed
import pickle
import torch

class LAIONImgembCaptionPairDataset(torch.utils.data.Dataset):

    def __init__(self, laion_data_dir):
        with open(os.path.join(laion_data_dir, 'augmented_image_embedding_dict.pkl'), 'rb') as f:
            imgemb_dict = pickle.load(f)

        with open(os.path.join(laion_data_dir, 'image_base_to_caption.pkl'), 'rb') as f:
            caption_dict = pickle.load(f)

        self.pairs = []
        self.image_bases = []
        for image_base in sorted(imgemb_dict.keys()):
            if image_base not in caption_dict:
                continue

            caption = caption_dict[image_base]
            if clip.tokenize(caption, truncate=True).squeeze().argmax().item() <= 1: #empty caption, only has start and end token
                continue

            self.pairs.append((imgemb_dict[image_base], caption_dict[image_base]))
            self.image_bases.append(image_base)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return {'imgemb' : self.pairs[idx][0], 'caption' : clip.tokenize(self.pairs[idx][1], truncate=True).squeeze(dim=0), 'idx' : torch.tensor(idx, dtype=torch.long)}
