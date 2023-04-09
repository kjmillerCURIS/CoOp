import os.path as osp
from collections import OrderedDict
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

#from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.engine import TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from data_manager_custom import DataManagerCustom
from custom_classification_evaluator import CustomClassificationEvaluator

import time
import datetime
from tqdm import tqdm

from .cocoop import PromptLearner
from .imagenet_templates import IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES_SELECT_ONETOKEN
from .zsclip import CUSTOM_TEMPLATES, CUSTOM_TEMPLATES_ONETOKEN

from .check_learnability import check_learnability

from write_to_log_file import write_to_log_file

_tokenizer = _Tokenizer()

LOW_MEMORY_MODE = True

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

#returns a list of pairs
#if random initialization, then it's a pair of numbers specifying number of tokens before and after class name
#if manual initialization, then it's a pair of strings specifying the initial state before and after class name
def get_ctx_inits(cfg):
    templates = IMAGENET_TEMPLATES_SELECT
    custom = CUSTOM_TEMPLATES
    if cfg.TRAINER.COCOOP_ENSEMBLING.ONETOKEN:
        templates = IMAGENET_TEMPLATES_SELECT_ONETOKEN
        custom = CUSTOM_TEMPLATES_ONETOKEN

    if cfg.DATASET.NAME != "ImageNet":
        templates += [custom[cfg.DATASET.NAME]]

    outputs = []
    for template in templates:
        assert('_' not in template)
        assert(template[-1] == '.')
        assert('{}' in template)
        parts = template[:-1].split('{}')
        assert(len(parts) == 2)
        n_ctx_or_ctx_init = []
        for part in parts:
            part = part.strip()
            if cfg.TRAINER.COCOOP_ENSEMBLING.RANDOM_CTX_INIT: #if random then put number of tokens
                n_ctx_or_ctx_init.append(len(_tokenizer.encode(part)))
            else: #if manual then put actual string
                n_ctx_or_ctx_init.append(part)

        outputs.append(tuple(n_ctx_or_ctx_init))

    return outputs

class CustomCLIPEnsembling(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner_list = []
        random_or_manual = {True : 'random', False : 'manual'}[cfg.TRAINER.COCOOP_ENSEMBLING.RANDOM_CTX_INIT]
        for n_ctx_or_ctx_init in get_ctx_inits(cfg):
            pl = PromptLearner(cfg, classnames, clip_model, random_or_manual, n_ctx_or_ctx_init, prec=cfg.TRAINER.COCOOP_ENSEMBLING.PREC, noperiod=cfg.TRAINER.COCOOP_ENSEMBLING.NOPERIOD)
            self.prompt_learner_list.append(pl)

        self.prompt_learner_list = nn.ModuleList(self.prompt_learner_list)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale #I don't think this ever gets updated, because it wouldn't have "prompt_learner" in its name
        self.dtype = clip_model.dtype

    #image_features should be normalized image features for whole batch
    #it should have shape (N, D)
    #returns normalized text features for whole batch
    #text_features will have shape (N, C, D) where C is number of classes
    def __compute_text_features(self, image_features, pl):
        tokenized_prompts = pl.tokenized_prompts
        (N, D) = image_features.shape
        C = tokenized_prompts.shape[0]
        prompts = pl(image_features)
        all_text_features = []
        for pts_i in prompts: #this is a loop across the batch
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_text_features.append(text_features)

        all_text_features = torch.stack(all_text_features)
        assert(all_text_features.shape == (N, C, D))
        return all_text_features

    def forward(self, image, label=None, train_separately=False):
        #Kevin's Note: Why on earth don't they do torch.no_grad??? Or is there a no_grad that I'm not aware of???
        #(but there's no point in changing it when it's a batch-size of 1)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_list = []
        for pl in self.prompt_learner_list:
            if LOW_MEMORY_MODE and self.prompt_learner_list[0].training: #this is just the first pass - we'll do this again
                with torch.no_grad():
                    text_features = self.__compute_text_features(image_features, pl)

                text_features.requires_grad = True
            else:
                text_features = self.__compute_text_features(image_features, pl)
            
            text_features_list.append(text_features)

        output = self.__forward_from_text_features_list(text_features_list, image_features, label=label, train_separately=train_separately)
        if not (LOW_MEMORY_MODE and self.prompt_learner_list[0].training): #in this case, just return loss or logits
            return output

        #ok, so at this point we're ready to do LOW_MEMORY_MODE backprop!
        assert(LOW_MEMORY_MODE and self.prompt_learner_list[0].training)

        #this backward() call will compute the grads of any parameters that came after the text features (currently none)
        #it will also compute grad w.r.t. text_features_list
        output.backward()

        #now we go through each tensor of that grad, use it to create a "fake" loss, and backprop from that loss
        #each iteration will compute the grads of a different prompt-learner, but it could just as well accumulate if needed
        for pl, text_features_for_grad in zip(self.prompt_learner_list, text_features_list):
            text_features_grad = text_features_for_grad.grad
            text_features = None #this *should* be the point where any memory between image_features and text_features can be garbage-collected
            text_features = self.__compute_text_features(image_features, pl)
            fake_loss = torch.sum(text_features * text_features_grad)
            fake_loss.backward() #this sets/accumulates any necessary gradients

        #make it non-differentiable so caller can't accidentally call backward() again
        return output.detach()

    #if we're training, this will give back the loss (yes, differentiable)
    #if we're testing, it'll give back the logits
    #this method does NOT know or care about LOW_MEMORY_MODE!
    #it's the caller's job to use the loss to get grad w.r.t. text_features_list and backprop from there in chunks, if that's what they wanna do
    #text_features_list should be list of tensors, each with shape (N, C, D)
    def __forward_from_text_features_list(self, text_features_list, image_features, label=None, train_separately=False):
        logit_scale = self.logit_scale.exp()
        if self.prompt_learner_list[0].training and train_separately:
            #compute losses separately, and average together
            losses = []
            for text_features in text_features_list:
                #text_features has shape (N, C, D)
                #image_features has shape (N, D)
                cossims = torch.bmm(text_features, torch.unsqueeze(image_features, 2)).squeeze(dim=2)
                logits = logit_scale * cossims
                loss = F.cross_entropy(logits, label)
                losses.append(loss)

            losses = torch.stack(losses)
            return torch.sum(losses)
        else:
            #ensemble text embeddings
            text_features = torch.mean(torch.stack(text_features_list), dim=0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            #compute logits
            #text_features has shape (N, C, D)
            #image_features has shape (N, D)
            cossims = torch.bmm(text_features, torch.unsqueeze(image_features, 2)).squeeze(dim=2)
            logits = logit_scale * cossims

            #return either logits or loss
            if self.prompt_learner_list[0].training:
                return F.cross_entropy(logits, label)
            else:
                return logits


#@TRAINER_REGISTRY.register()
class CoCoOpEnsembling(TrainerX):
    
    def __init__(self, cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type):
        assert(train_or_test in ['train', 'test'])
        
        #stuff from TrainerBase.__init__()
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        
        #stuff from SimpleTrainer.__init__(), but use train_or_test for some things
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader(fewshot_seed, domain_split_index, class_split_type, eval_type)
        self.build_model(train_or_test)
        if train_or_test == 'test': #no need to run evaluator if we're not testing!
            self.evaluator = CustomClassificationEvaluator(cfg, self.lab2cname_test)
        else:
            self.evaluator = None

        self.best_result = -np.inf
    
    def build_data_loader(self, fewshot_seed, domain_split_index, class_split_type, eval_type):
        """Create essential data-related attributes.
        """

        dm = DataManagerCustom(self.cfg, fewshot_seed, domain_split_index, class_split_type, eval_type)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes_train, self.num_classes_test = dm.num_classes_train, dm.num_classes_test
        self.num_source_domains = dm.num_source_domains
        self.lab2cname_train, self.lab2cname_test = dm.lab2cname_train, dm.lab2cname_test  # dict {label: classname}

        self.dm = dm
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP_ENSEMBLING.PREC in ["fp16", "fp32", "amp"]

    def build_model(self, train_or_test):
        assert(train_or_test in ['train', 'test'])
        cfg = self.cfg
        classnames = {'train' : self.dm.dataset.classnames_train, 'test' : self.dm.dataset.classnames_test}[train_or_test]

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP_ENSEMBLING.PREC == "fp32" or cfg.TRAINER.COCOOP_ENSEMBLING.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIPEnsembling(cfg, classnames, clip_model) #training-vs-testing-classnames problem has already been taken care of

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner_list"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                assert('prompt_learner' not in name)
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        check_learnability(enabled, ['meta_net', 'ctx_beforename', 'ctx_aftername'])

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner_list, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner_list, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner_list", self.model.prompt_learner_list, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP_ENSEMBLING.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        train_separately = self.cfg.TRAINER.COCOOP_ENSEMBLING.TRAIN_SEPARATELY
        
        prec = self.cfg.TRAINER.COCOOP_ENSEMBLING.PREC
        if prec == "amp":
            assert(not LOW_MEMORY_MODE) #LOW_MEMORY_MODE doesn't support mixed precision (for now)
            with autocast():
                loss = model(image, label, train_separately=train_separately)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            optim.zero_grad() #do this first cuz model.forward() might also call backward()
            loss = model(image, label, train_separately=train_separately)
#            write_to_log_file(str(time.time()))
            if not LOW_MEMORY_MODE: #if LOW_MEMORY_MODE then model.forward() already did a backward() call
                loss.backward()

            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            
            for k in sorted(state_dict.keys()):
                if 'DONOTLOAD' in k:
                    del state_dict[k]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        details = {}
        
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
            output = self.model(input)
            self.evaluator.process(output, label, domain)
            test_loss = F.cross_entropy(output, label, reduce=False, reduction='none')
            for i, impath in enumerate(batch['impath']):
                detail = {'test_loss': test_loss[i].item()}
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
    
    def after_train(self):
        print("Finish training")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()
