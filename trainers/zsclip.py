import os.path as osp
from collections import OrderedDict
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

#from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.engine import TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from data_manager_custom import DataManagerCustom
from custom_classification_evaluator import CustomClassificationEvaluator

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES_SELECT_ONETOKEN

import time
import datetime
from tqdm import tqdm

from .exposed_text_encoder import ExposedTransformer


CUSTOM_TEMPLATES = {
    "DomainNetCustom": "a photo of a {}.",
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

CUSTOM_TEMPLATES_ONETOKEN = {"DomainNetCustom": "photo {}."}

class ExposedTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.transformer = ExposedTransformer(clip_model.transformer)
    
    def encode_text(self, text):
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn_probs = self.transformer(x, cls_indices=text.argmax(dim=-1))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection

        return x, attn_probs

#@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    
    def __init__(self, cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type, record_attentropy=False):
        self.record_attentropy = record_attentropy
        assert(train_or_test == 'test')
        
        #stuff from TrainerBase.__init__()
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        
        #stuff from SimpleTrainer.__init__(), but use train_or_test for some things
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader(fewshot_seed, domain_split_index, class_split_type, eval_type)
        self.build_model()
        self.evaluator = CustomClassificationEvaluator(cfg, self.lab2cname_test)

        self.best_result = -np.inf
    
    def build_data_loader(self, fewshot_seed, domain_split_index, class_split_type, eval_type):
        """Create essential data-related attributes.
        """

        dm = DataManagerCustom(self.cfg, fewshot_seed, domain_split_index, class_split_type, eval_type)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes_test = dm.num_classes_test
        self.lab2cname_test = dm.lab2cname_test  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames_test

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        if self.record_attentropy:
            self.text_encoder = ExposedTextEncoder(clip_model)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            if self.record_attentropy:
                text_features, attn_probs = self.text_encoder.encode_text(prompts)
                assert(prompts.shape[0] == len(classnames))
                assert(attn_probs.shape == (len(classnames), prompts.shape[1]))
                attn_probs = torch.clamp(attn_probs, min=1e-8)
                self.attentropies = torch.sum(-attn_probs * torch.log(attn_probs), dim=1).cpu().numpy()
                assert(self.attentropies.shape == (len(classnames),))
            else:
                text_features = clip_model.encode_text(prompts)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

    @torch.no_grad()
    def test(self, split=None):
        details = {}

        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label, domain)
            test_loss = F.cross_entropy(output, label, reduce=False, reduction='none')
            for i, impath in enumerate(batch['impath']):
                detail = {'test_loss': test_loss[i].item()}
                if self.record_attentropy:
                    detail['attentropies'] = self.attentropies
                    detail['correct_attentropy'] = self.attentropies[label[i].item()]
                    detail['pred_attentropy'] = self.attentropies[output[i,:].argmax().item()]
                    detail['avg_attentropy'] = np.mean(self.attentropies)

                details[impath] = detail

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        results['details'] = details

        return list(results.values())[0], results
    
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain


#@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def __init__(self, cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type):
       super().__init__(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type)

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames_test

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        if self.cfg.TRAINER.ZEROSHOTCLIP2.ONETOKEN:
            self.templates = IMAGENET_TEMPLATES_SELECT_ONETOKEN

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            if self.cfg.TRAINER.ZEROSHOTCLIP2.ONETOKEN:
                self.templates += [CUSTOM_TEMPLATES_ONETOKEN[cfg.DATASET.NAME]]
            else:
                self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model
