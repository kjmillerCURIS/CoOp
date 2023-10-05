import os
import sys
import pickle
import torch
from tqdm import tqdm
from laion_imgemb_caption_pair_dataset import LAIONImgembCaptionPairDataset

BASE_DIR = '../vislang-domain-exploration-data/CoCoOpExperiments'
LAION_DATA_DIR = os.path.join(BASE_DIR, 'laion_data/uniform_subset')
MODEL_FILENAME = os.path.join(LAION_DATA_DIR, 'CLIP_OCR_model.pkl')
OUT_FILENAME = os.path.join(LAION_DATA_DIR, 'laion_without_text.pkl')

NUM_WORKERS = 2
BATCH_SIZE = 1024

def filter_text_from_laion(laion_data_dir, model_filename, out_filename):
    dataset = LAIONImgembCaptionPairDataset(laion_data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=False, shuffle=False)
    with open(model_filename, 'rb') as f:
        my_clf = pickle.load(f)

    textless_images = set([])
    for batch in tqdm(dataloader):
        embs = batch['imgemb']
        embs = embs / embs.norm(dim=-1, keepdim=True)
        preds = my_clf.predict(embs.numpy())
        for idx, pred in zip(batch['idx'], preds):
            if pred == 0:
                textless_images.add(dataset.image_bases[idx.item()])

    with open(out_filename, 'wb') as f:
        pickle.dump(textless_images, f)

if __name__ == '__main__':
    filter_text_from_laion(LAION_DATA_DIR, MODEL_FILENAME, OUT_FILENAME)
