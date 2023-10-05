import os
import sys
import clip #clip.tokenize() just uses SimpleTokenizer and then zero-pads and/or truncates as needed
import pickle
import torch

FULL_CONTEXT_LENGTH = 77

class LAIONImgembClassnamePairDataset(torch.utils.data.Dataset):

    def __init__(self, laion_data_dir, gramlen, laion_without_text_filename=None):
        with open(os.path.join(laion_data_dir, 'augmented_image_embedding_dict.pkl'), 'rb') as f:
            imgemb_dict = pickle.load(f)

        with open(os.path.join(laion_data_dir, 'caption_classname_dict.pkl'), 'rb') as f:
            caption_classname_dict = pickle.load(f)

        if laion_without_text_filename is not None:
            with open(laion_without_text_filename, 'rb') as f:
                laion_without_text = pickle.load(f)

        self.pairs = []
        self.image_bases = []
        for image_base in sorted(imgemb_dict.keys()):
            if image_base not in caption_classname_dict:
                continue

            if laion_without_text_filename is not None:
                if image_base not in laion_without_text:
                    continue

            classname_candidates = caption_classname_dict[image_base]
            best_classname = None
            best_cossim = float('-inf')
            for k in sorted(classname_candidates.keys()):
                if k > gramlen:
                    continue

                if classname_candidates[k][1] > best_cossim:
                    best_cossim = classname_candidates[k][1]
                    best_classname = classname_candidates[k][0]

            self.pairs.append((imgemb_dict[image_base], best_classname))
            self.image_bases.append(image_base)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return {'imgemb' : self.pairs[idx][0], 'classname' : torch.cat([torch.tensor(self.pairs[idx][1], dtype=torch.long), torch.zeros(FULL_CONTEXT_LENGTH - self.pairs[idx][1].shape[0], dtype=torch.long)], dim=0)}
