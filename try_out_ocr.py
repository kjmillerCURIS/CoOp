import os
import sys
import cv2
import glob
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm
from caption_image import caption_image

IMAGE_BLOWUP = 2
FONT_SIZE_MULTIPLIER = 14 / 600
MIN_FONT_SIZE = 14

BASE_DIR = '../vislang-domain-exploration-data/CoCoOpExperiments/EDA/TryOutOCR'
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
VIS_DIR = os.path.join(BASE_DIR, 'vis')

OCR_ID_LIST = ['microsoft/trocr-base-printed','microsoft/trocr-base-handwritten','microsoft/trocr-large-printed','microsoft/trocr-large-handwritten']

def load_image(image_path):
    numI = cv2.imread(image_path)
    if IMAGE_BLOWUP != 1:
        numI = cv2.resize(numI, None, fx=IMAGE_BLOWUP, fy=IMAGE_BLOWUP)

    return numI

def try_out_ocr_helper(image_dir, vis_dir, ocr_id, subdir):
    processor = TrOCRProcessor.from_pretrained(ocr_id)
    model = VisionEncoderDecoderModel.from_pretrained(ocr_id).to('cuda')

    in_dir = os.path.join(image_dir, subdir)
    images = sorted(glob.glob(os.path.join(in_dir, '*.jpg')))

    out_dir = os.path.join(vis_dir, ocr_id.replace('/', '__'), subdir)
    os.makedirs(out_dir, exist_ok=True)

    for image in tqdm(images):
        img = Image.open(image)
        if img.mode != 'RGB':
            print('not RGB "%s"'%(image))
            continue

        pixel_values = processor(img, return_tensors='pt').pixel_values.to('cuda')
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids.to('cpu'), skip_special_tokens=True)[0]
        generated_text = generated_text.strip()
        numI = load_image(image)
        numIvis = caption_image(numI, [generated_text if generated_text != '' else '[NO TEXT DETECTED]'], font_size=max(int(round(FONT_SIZE_MULTIPLIER * numI.shape[1])), MIN_FONT_SIZE))
        vis_filename = os.path.join(out_dir, os.path.splitext(os.path.basename(image))[0] + '.png')
        cv2.imwrite(vis_filename, numIvis)

def try_out_ocr(image_dir, vis_dir):
    for ocr_id in OCR_ID_LIST:
        for subdir in ['pos', 'neg']:
            try_out_ocr_helper(image_dir, vis_dir, ocr_id, subdir)

if __name__ == '__main__':
    try_out_ocr(IMAGE_DIR, VIS_DIR)
