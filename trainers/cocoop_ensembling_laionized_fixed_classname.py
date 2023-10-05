import os.path as osp
from collections import OrderedDict
import math
import numpy as np
import random

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
from .efficient_text_encoder import EfficientTextEncoder, EFFICIENT_CONTEXT_LENGTH
from .imagenet_templates import IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES_SELECT_ONETOKEN
from .zsclip import CUSTOM_TEMPLATES, CUSTOM_TEMPLATES_ONETOKEN
from laion_imgemb_classname_pair_dataset import LAIONImgembClassnamePairDataset

from .check_learnability import check_learnability

from write_to_log_file import write_to_log_file

_tokenizer = _Tokenizer()

NUM_LAION_WORKERS = 2
LAION_WITHOUT_TEXT_BASE = 'laion_without_text.pkl'

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


#returns a list of pairs
#if random initialization, then it's a pair of numbers specifying number of tokens before and after class name
#if manual initialization, then it's a pair of strings specifying the initial state before and after class name
def get_ctx_inits(cfg):
    templates = IMAGENET_TEMPLATES_SELECT
    custom = CUSTOM_TEMPLATES
    if cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.ONETOKEN:
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
            if cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.RANDOM_CTX_INIT: #if random then put number of tokens
                n_ctx_or_ctx_init.append(len(_tokenizer.encode(part)))
            else: #if manual then put actual string
                n_ctx_or_ctx_init.append(part)

        outputs.append(tuple(n_ctx_or_ctx_init))

    return outputs


class PromptLearnerLAIONizedFixedClassname(PromptLearner):
    def __init__(self, cfg, classnames, clip_model, random_or_manual, n_ctx_or_ctx_init, prec='fp16', noperiod=False):

        #setup all the stuff for prompt production
        super().__init__(cfg, classnames, clip_model, random_or_manual, n_ctx_or_ctx_init, prec=prec, noperiod=noperiod, num_meta_out=1)

        #setup stuff for making prompts for LAION
        #set it up for a zero-token classname and adjust later as needed
        LAION_str = ''
        end_offset = 1
        if self.ctx_beforename is not None:
            LAION_str = LAION_str + ' '.join(['X' for i in range(self.ctx_beforename.shape[0])]) + ' '
            end_offset += self.ctx_beforename.shape[0]

        classname_offset = end_offset #this is how we split tokenized_LAION_prompts into part before and after classname
        if self.ctx_aftername is not None:
            LAION_str = LAION_str + ' ' + ' '.join(['X' for i in range(self.ctx_aftername.shape[0])])
            end_offset += self.ctx_aftername.shape[0]

        if not noperiod:
            LAION_str = LAION_str + '.'

        tokenized_LAION_prompt = clip.tokenize(LAION_str)[0]
        self.tokenized_LAION_prompt_before_classname = tokenized_LAION_prompt[:classname_offset].cuda()
        self.tokenized_LAION_prompt_after_classname = tokenized_LAION_prompt[classname_offset:].cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_LAION_prompt).type(clip_model.dtype)

        self.register_buffer('DONOTLOAD_LAION_start_part', embedding[:1,:]) #SOS token
        self.register_buffer('DONOTLOAD_LAION_end_part', embedding[end_offset:,:]) #period and EOS and padding

    #expect classname_tokens to have shape (n_cls, n_tkn, tkn_dim)
    #expect tokenized_classnames to have shape (n_cls, n_tkn)
    #expect im_features to have shape (n_cls, vis_dim)
    #assume that im_features is already normalized
    #returns prompts, tokenized_LAION_prompts (this is different from other PromptLearner varieties which only return the prompts tensor)
    #prompts will have shape (len(positive_indices), n_cls, n_tkn, tkn_dim)
    #tokenized_LAION_prompts will have shape (n_cls, n_tkn)
    def forward_for_LAION(self, classname_tokens, tokenized_classnames, im_features, positive_indices):

        assert(im_features.shape[0] == self.n_cls)
        assert(tokenized_classnames.shape[0] == im_features.shape[0])
        assert(classname_tokens.shape[0] == im_features.shape[0])

        #just get the meta-token (that's all there is)
        bias = self.compute_meta_token(im_features[positive_indices,:])
        bias = bias.unsqueeze(1) #(len(positive_indices), 1, tkn_dim)
        if self.ctx_beforename is not None:
            ctx_beforename_shifted = self.ctx_beforename.unsqueeze(0) + bias

        if self.ctx_aftername is not None:
            ctx_aftername_shifted = self.ctx_aftername.unsqueeze(0) + bias

        #first we get the name_parts
        #we assume that tokenized_classnames has an SOS token (and classname_tokens has the embedding of it)
        name_lens = tokenized_classnames.argmax(dim=1) - 1 #yes this is correct, just think about it
        name_parts = [classname_tokens_i[1:1+name_len.item()] for classname_tokens_i, name_len in zip(classname_tokens, name_lens)]

        #now we get start_parts, which is straightforward :)
        start_parts = [self._buffers['DONOTLOAD_LAION_start_part'] for i in range(self.n_cls)]

        #now for end_parts, chop off whatever the classname takes up
        full_end_len = self._buffers['DONOTLOAD_LAION_end_part'].shape[0]
        end_parts = [self._buffers['DONOTLOAD_LAION_end_part'][:full_end_len - name_len.item()] for name_len in name_lens]

        #now make the prompts!
        prompts = []
        for i in range(len(positive_indices)): #loop over positive indices
            ctx_beforename_i = None
            if self.ctx_beforename is not None:
                ctx_beforename_i = ctx_beforename_shifted[i,:,:]

            ctx_aftername_i = None
            if self.ctx_aftername is not None:
                ctx_aftername_i = ctx_aftername_shifted[i,:,:]

            pts_i = self.construct_prompts_helper(start_parts, ctx_beforename_i, name_parts, ctx_aftername_i, end_parts)
            assert(pts_i.shape[0] == self.n_cls)
            prompts.append(pts_i)

        prompts = torch.stack(prompts).type(torch.float16) #not sure why it wasn't already torch.float16...
        assert(len(prompts.shape) == 4)
        assert(prompts.shape[:2] == (len(positive_indices), self.n_cls))

        #now make tokenized_LAION_prompts
        full_after_len = self.tokenized_LAION_prompt_after_classname.shape[0]
        tokenized_LAION_prompts = [torch.cat([self.tokenized_LAION_prompt_before_classname, tokenized_classname[1:1+name_len.item()], self.tokenized_LAION_prompt_after_classname[:full_after_len - name_len.item()]], dim=0) for tokenized_classname, name_len in zip(tokenized_classnames, name_lens)]
        tokenized_LAION_prompts = torch.stack(tokenized_LAION_prompts, dim=0)

        #prompts will have shape (len(positive_indices), n_cls, n_tkn, tkn_dim)
        #tokenized_LAION_prompts will have shape (n_cls, n_tkn)
        assert(prompts.shape[:2] == (len(positive_indices), self.n_cls))
        assert(tokenized_LAION_prompts.shape == prompts.shape[1:3])

        return prompts, tokenized_LAION_prompts


class CustomCLIPEnsemblingLAIONizedFixedClassname(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner_list = []
        random_or_manual = {True : 'random', False : 'manual'}[cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.RANDOM_CTX_INIT]
        for n_ctx_or_ctx_init in get_ctx_inits(cfg):
            pl = PromptLearnerLAIONizedFixedClassname(cfg, classnames, clip_model, random_or_manual, n_ctx_or_ctx_init, prec=cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.PREC, noperiod=cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.NOPERIOD)
            needed_length = pl.tokenized_prompts.argmax(dim=-1).max().item()
            print('needed_length=%d'%(needed_length))
            assert(needed_length < EFFICIENT_CONTEXT_LENGTH)
            self.prompt_learner_list.append(pl)

        self.prompt_learner_list = nn.ModuleList(self.prompt_learner_list)
        self.image_encoder = clip_model.visual
        self.text_encoder = EfficientTextEncoder(clip_model)
        self.token_embedding = clip_model.token_embedding #this is safe because it is NOT owned by PromptLearner, therefore it will NOT becomoe learnable!
        self.logit_scale = clip_model.logit_scale #I don't think this ever gets updated, because it wouldn't have "prompt_learner" in its name
        self.dtype = clip_model.dtype
        self.cfg = cfg

    def forward(self, image, label=None, train_separately=False):
        #Kevin's Note: Why on earth don't they do torch.no_grad??? Or is there a no_grad that I'm not aware of???
        #(but there's no point in changing it when it's a batch-size of 1)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts_list = []
        for pl in self.prompt_learner_list:
            prompts = pl(image_features) #(batch, n_cls, n_tkn, tkn_dim)
            prompts = prompts[:,:,:EFFICIENT_CONTEXT_LENGTH,:] #cut out needless padding to make it faster!
            prompts_list.append(prompts)

        prompts = torch.stack(prompts_list, dim=1) #(batch, n_comp, n_cls, n_tkn, tkn_dim)
        prompts = torch.flatten(prompts, start_dim=1, end_dim=2) #(batch, n_comp * n_cls, n_tkn, tkn_dim)
        tokenized_prompts = torch.cat([pl.tokenized_prompts[:,:EFFICIENT_CONTEXT_LENGTH] for pl in self.prompt_learner_list], dim=0)

        #run the text backbone
        text_features = []
        for pts_i in prompts: #this is a loop across the batch
            text_features_i = self.text_encoder(pts_i, tokenized_prompts)
            text_features_i = text_features_i / text_features_i.norm(dim=-1, keepdim=True)
            text_features.append(text_features_i)

        text_features = torch.stack(text_features) #(batch, n_comp * n_cls, vis_dim)
        text_features = torch.unflatten(text_features, 1, (len(self.prompt_learner_list), self.prompt_learner_list[0].n_cls)) #(batch, n_comp, n_cls, vis_dim)
        text_features = text_features.permute(1, 0, 2, 3) #(n_comp, batch, n_cls, vis_dim)
        output = self.__forward_from_text_features_list(text_features, image_features, label=label, train_separately=train_separately)
        return output

    #remember, this one is called ONLY during training! (because LAION is used only during training)
    def forward_for_LAION(self, tokenized_classnames, image_features, train_separately=False):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        #embed the captions
        with torch.no_grad():
            classname_tokens = self.token_embedding(tokenized_classnames).type(self.dtype)

        if not train_separately: #same positive_indices for all components
            positive_indices = random.sample(range(image_features.shape[0]), self.cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.LAION_NUM_POSITIVES)
        else:
            positive_indices_list = []

        prompts_list = []
        tokenized_LAION_prompts_list = []
        for pl in self.prompt_learner_list:
            if train_separately: #different positive_indices for each component
                positive_indices = random.sample(range(image_features.shape[0]), self.cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.LAION_NUM_POSITIVES)
                positive_indices_list.append(positive_indices)

            prompts, tokenized_LAION_prompts = pl.forward_for_LAION(classname_tokens, tokenized_classnames, image_features, positive_indices)
            prompts = prompts[:,:,:EFFICIENT_CONTEXT_LENGTH,:] #cut out needless padding to make it faster!
            tokenized_LAION_prompts = tokenized_LAION_prompts[:,:EFFICIENT_CONTEXT_LENGTH]
            prompts_list.append(prompts)
            tokenized_LAION_prompts_list.append(tokenized_LAION_prompts)

        prompts = torch.stack(prompts_list, dim=1) #(len(positive_indices), n_comp, n_cls, n_tkn, tkn_dim)
        prompts = torch.flatten(prompts, start_dim=1, end_dim=2) #(len(positive_indices), n_comp * n_cls, n_tkn, tkn_dim)
        tokenized_LAION_prompts = torch.cat(tokenized_LAION_prompts_list, dim=0) #(n_comp * n_cls, n_tkn)

        #run the text backbone
        text_features = []
        for pts_i in prompts: #this is a loop across the positive_indices
            text_features_i = self.text_encoder(pts_i, tokenized_LAION_prompts)
            text_features_i = text_features_i / text_features_i.norm(dim=-1, keepdim=True)
            text_features.append(text_features_i)

        text_features = torch.stack(text_features) #(len(positive_indices), n_comp * n_cls, vis_dim)
        text_features = torch.unflatten(text_features, 1, (len(self.prompt_learner_list), self.prompt_learner_list[0].n_cls)) #(len(positive_indices), n_comp, n_cls, vis_dim)
        text_features = text_features.permute(1, 0, 2, 3) #(n_comp, len(positive_indices), n_cls, vis_dim)

        #prepare image features and labels for next step
        if train_separately:
            image_features_for_clf = [image_features[idx,:] for idx in positive_indices_list]
            label_for_clf = [torch.tensor(y).cuda() for y in positive_indices_list]
        else:
            image_features_for_clf = image_features[positive_indices,:]
            label_for_clf = torch.tensor(positive_indices).cuda()

        output = self.__forward_from_text_features_list(text_features, image_features_for_clf, label=label_for_clf, train_separately=train_separately, LAION_mode=True)
        return output

    #if we're training, this will give back the loss (yes, differentiable)
    #if we're testing, it'll give back the logits
    #text_features_list should be list of tensors, each with shape (batch, n_cls, vis_dim), or a single tensor with shape (n_comp,batch,n_cls,vis_dim)
    #if LAION_mode, then image_features and label will be lists of tensors if train_separately, otherwise just tensors as usual
    def __forward_from_text_features_list(self, text_features_list, image_features, label=None, train_separately=False, LAION_mode=False):
        if LAION_mode:
            assert(self.prompt_learner_list[0].training)
            assert(label is not None)

        logit_scale = self.logit_scale.exp()
        if self.prompt_learner_list[0].training and train_separately:
            #compute losses separately, and sum together
            losses = []
            if LAION_mode:
                for text_features, image_features_one_comp, label_one_comp in zip(text_features_list, image_features, label):
                    #text_features has shape (N, C, D)
                    #image_features_one_comp has shape (N, D)
                    cossims = torch.bmm(text_features, torch.unsqueeze(image_features_one_comp, 2)).squeeze(dim=2)
                    logits = logit_scale * cossims
                    loss = F.cross_entropy(logits, label_one_comp)
                    losses.append(loss)
            else:
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
            #in this case the same code can handle LAION_mode==True and LAION_mode==False
            text_features_stacked = text_features_list
            if isinstance(text_features_stacked, list):
                text_features_stacked = torch.stack(text_features_stacked)

            text_features = torch.mean(text_features_stacked, dim=0)
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


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


#@TRAINER_REGISTRY.register()
class CoCoOpEnsemblingLAIONizedFixedClassname(TrainerX):
    
    def __init__(self, cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type, laion_data_dir):
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

        if train_or_test == 'train':
            laion_without_text_filename = None
            if cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.FILTER_OUT_TEXT_ON_LAION_IMAGES:
                laion_without_text_filename = osp.join(laion_data_dir, LAION_WITHOUT_TEXT_BASE)

            laion_dataset = LAIONImgembClassnamePairDataset(laion_data_dir, cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.GRAMLEN, laion_without_text_filename=laion_without_text_filename)
            laion_dataloader = torch.utils.data.DataLoader(laion_dataset, batch_size=self.num_classes_train, shuffle=True, drop_last=True, num_workers=NUM_LAION_WORKERS)
            self.laion_genny = infinite_dataloader(laion_dataloader)

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
        assert cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.PREC in ["fp16", "fp32", "amp"]

    def build_model(self, train_or_test):
        assert(train_or_test in ['train', 'test'])
        cfg = self.cfg
        classnames = {'train' : self.dm.dataset.classnames_train, 'test' : self.dm.dataset.classnames_test}[train_or_test]

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.PREC == "fp32" or cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIPEnsemblingLAIONizedFixedClassname(cfg, classnames, clip_model) #training-vs-testing-classnames problem has already been taken care of

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

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        laion_batch = next(self.laion_genny)
        laion_image_features, laion_tokenized_classnames = laion_batch['imgemb'].cuda(), laion_batch['classname'].cuda()

        model = self.model
        optim = self.optim
        scaler = self.scaler
        train_separately = self.cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.TRAIN_SEPARATELY

        prec = self.cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.PREC
        if prec == "amp":
            optim.zero_grad()
            with autocast():
                loss = model(image, label, train_separately=train_separately)
            scaler.scale(loss).backward()
            with autocast():
                laion_loss = model.forward_for_LAION(laion_tokenized_classnames, laion_image_features, train_separately=train_separately)
                laion_loss_for_backward = self.cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.LAION_LAMBDA * laion_loss

            scaler.scale(laion_loss_for_backward).backward()
            scaler.step(optim)
            scaler.update()
        else:
            optim.zero_grad()
            loss = model(image, label, train_separately=train_separately)
            loss.backward()
            laion_loss = model.forward_for_LAION(laion_tokenized_classnames, laion_image_features, train_separately=train_separately)
            laion_loss_for_backward = self.cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED_FIXED_CLASSNAME.LAION_LAMBDA * laion_loss
            laion_loss_for_backward.backward()
            optim.step()

        total_loss = loss.detach() + laion_loss_for_backward.detach()
        assert(not np.isnan(total_loss.item()))
        loss_summary = {'main_loss': loss.item(), 'laion_loss' : laion_loss.item(), 'loss' : total_loss.item()}

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
