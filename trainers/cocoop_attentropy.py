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

from data_manager_custom import DataManagerCustom
from custom_classification_evaluator import CustomClassificationEvaluator

import time
import datetime
from tqdm import tqdm

from .cocoop import CustomCLIP, load_clip_to_cpu

from .check_learnability import check_learnability


#@TRAINER_REGISTRY.register()
class CoCoOpAttentropy(TrainerX):
    
    def __init__(self, cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type, record_attentropy=False):
        assert(train_or_test in ['train', 'test'])
        self.record_attentropy = record_attentropy
        
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
        assert cfg.TRAINER.COCOOP_ATTENTROPY.PREC in ["fp16", "fp32", "amp"]

    def build_model(self, train_or_test):
        assert(train_or_test in ['train', 'test'])
        cfg = self.cfg
        classnames = {'train' : self.dm.dataset.classnames_train, 'test' : self.dm.dataset.classnames_test}[train_or_test]

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP_ATTENTROPY.PREC == "fp32" or cfg.TRAINER.COCOOP_ATTENTROPY.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, cfg.TRAINER.COCOOP_ATTENTROPY, classnames, clip_model, (train_or_test == 'train' or self.record_attentropy)) #training-vs-testing-classnames problem has already been taken care of

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
         
        print(f"Parameters to be updated: {enabled}")
        check_learnability(enabled, ['meta_net', 'ctx_beforename', 'ctx_aftername'])

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP_ATTENTROPY.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    #this will average entropies across all the classes, and then across the batch
    #(this is entropy of a distribution of token indices)
    def __compute_attentropy_loss(self, attn_probs):
        assert(len(attn_probs.shape) == 3)
        attn_probs = torch.clamp(attn_probs, min=1e-8)
        attentropies = torch.sum(-attn_probs * torch.log(attn_probs), dim=2)
        avg_attentropy = torch.mean(attentropies) #average over classes and batch, all at once!
        return -avg_attentropy #want to maximize attentropy

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.COCOOP_ATTENTROPY.PREC
        if prec == "amp":
            with autocast():
                xent_loss, attn_probs = model(image, label)
                loss = xent_loss + self.cfg.TRAINER.COCOOP_ATTENTROPY.ATTENTROPY_LAMBDA * self.__compute_attentropy_loss(attn_probs)

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            xent_loss, attn_probs = model(image, label)
            loss = xent_loss + self.cfg.TRAINER.COCOOP_ATTENTROPY.ATTENTROPY_LAMBDA * self.__compute_attentropy_loss(attn_probs)
            optim.zero_grad()
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
            if self.record_attentropy:
                output, attn_probs = self.model(input)
                assert(len(attn_probs.shape) == 3)
                assert(attn_probs.shape[0] == input.shape[0])
                assert(attn_probs.shape[1] == self.num_classes_test)
                attn_probs = torch.clamp(attn_probs, min=1e-8)
                attentropies = torch.sum(-attn_probs * torch.log(attn_probs), dim=2)
                correct_attentropy = attentropies[torch.arange(attentropies.shape[0]), label].cpu().numpy()
                pred_attentropy = attentropies[torch.arange(attentropies.shape[0]), output.argmax(dim=1)].cpu().numpy()
                avg_attentropy = torch.mean(attentropies, dim=1).cpu().numpy()
                attentropies = attentropies.cpu().numpy()
            else:
                output = self.model(input)

            self.evaluator.process(output, label, domain)
            test_loss = F.cross_entropy(output, label, reduce=False, reduction='none')
            for i, impath in enumerate(batch['impath']):
                detail = {'test_loss': test_loss[i].item()}
                if self.record_attentropy:
                    detail['attentropies'] = attentropies[i,:]
                    detail['correct_attentropy'] = correct_attentropy[i]
                    detail['pred_attentropy'] = pred_attentropy[i]
                    detail['avg_attentropy'] = avg_attentropy[i]

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
