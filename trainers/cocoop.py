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

from .exposed_text_encoder import ExposedTextEncoder

_tokenizer = _Tokenizer()


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

#prompt will be "[SOS][LEARNABLE1][CLASS NAME][LEARNABLE2][PERIOD][EOS]"
#LEARNABLE1 and LEARNABLE2 will be the parts that are learned/generated
#note that "EOS" is the one that becomes the text embedding (you might've also called it the "CLS token" in your head)
class PromptLearner(nn.Module):
    
    #n_ctx and ctx_init should both be pairs
    #use empty strings for empty context, and None for random init (BOTH must be None)
    def __init__(self, cfg, classnames, clip_model, n_ctx, ctx_init):
        super().__init__()
        n_ctx_beforename, n_ctx_aftername = n_ctx
        ctx_init_beforename, ctx_init_aftername = ctx_init
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init_beforename or ctx_init_aftername:
            assert(ctx_init_beforename is not None and ctx_init_aftername is not None)
            # use given words to initialize context vectors
            ctx_init_beforename = ctx_init_beforename.replace('_', ' ')
            ctx_init_aftername = ctx_init_aftername.replace('_', ' ')
            assert(len(ctx_init_beforename.split(' ')) * int(ctx_init_beforename != '') == n_ctx_beforename)
            assert(len(ctx_init_aftername.split(' ')) * int(ctx_init_aftername != '') == n_ctx_aftername)
            if ctx_init_beforename != '':
                prompt = clip.tokenize(ctx_init_beforename)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)

                ctx_vectors = embedding[0, 1:1+n_ctx_beforename, :]
                self.ctx_beforename = nn.Parameter(ctx_vectors)
                prompt_filler_beforename = ctx_init_beforename + ' '
            else:
                self.ctx_beforename = None
                prompt_filler_beforename = ''
            
            if ctx_init_aftername != '':
                prompt = clip.tokenize(ctx_init_aftername)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)

                ctx_vectors = embedding[0, 1:1+n_ctx_aftername, :]
                self.ctx_aftername = nn.Parameter(ctx_vectors)
                prompt_filler_aftername = ' ' + ctx_init_aftername
            else:
                self.ctx_aftername = None
                prompt_filler_aftername = ''
        
        else:
            assert(ctx_init_beforename is None and ctx_init_aftername is None)
            # random initialization
            if n_ctx_beforename > 0:
                ctx_vectors = torch.empty(n_ctx_beforename, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.ctx_beforename = nn.Parameter(ctx_vectors)
                prompt_filler_beforename = ' '.join(['X'] * n_ctx_beforename) + ' '
            else:
                self.ctx_beforename = None
                prompt_filler_beforename = ''
            
            if n_ctx_aftername > 0:
                ctx_vectors = torch.empty(n_ctx_aftername, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.ctx_aftername = nn.Parameter(ctx_vectors)
                prompt_filler_aftername = ' ' + ' '.join(['X'] * n_ctx_aftername)
            else:
                self.ctx_aftername = None
                prompt_filler_aftername = ''

        print(f'Initial contexts: "{prompt_filler_beforename}", "{prompt_filler_aftername}"')
        print(f"Number of context words (tokens): {n_ctx_beforename}, {n_ctx_aftername}")

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_filler_beforename + name + prompt_filler_aftername + '.' for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        #have to store things as lists because some class names might have different numbers of tokens, so parts might have different lengths
        self.start_parts, self.name_parts, self.end_parts = [], [], []
        for i, name, name_len in enumerate(zip(classnames, name_lens)):
            self.start_parts.append(embedding[i, :1, :]) #SOS token
            self.name_parts.append(embedding[i, 1+n_ctx_beforename:1+n_ctx_beforename+name_len, :]) #classname tokens
            self.end_parts.append(embedding[i, 1+n_ctx_beforename+name_len+n_ctx_aftername:, :]) #period and EOS and padding

        self.n_cls = n_cls
        self.n_ctx_beforename = n_ctx_beforename
        self.n_ctx_aftername = n_ctx_aftername
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx_beforename, ctx_aftername):
        #this assumes that we're constructing one prompt for each class
        #ctx_beforename should have shape (n_ctx_beforename, ctx_dim)
        #ctx_aftername should have shape (n_ctx_aftername, ctx_dim)
        #self.start_parts, self.name_parts, self.end_parts will be lists of (*, ctx_dim) tensors
        #self.end_parts would have the period and the EOS token, and a bunch of padding (which the transformer will ignore cuz it's causal)

        prompts = []
        for start_part, name_part, end_part in zip(self.start_parts, self.name_parts, self.end_parts):
            items = [start_part]
            if ctx_beforename is not None:
                items.append(ctx_beforename)

            items.append(name_part)
            if ctx_aftername is not None:
                items.append(ctx_aftername)

            items.append(end_part)
            prompt = torch.cat(items, dim=0)
            prompts.append(prompt)

        return torch.stack(prompts)

    def forward(self, im_features):
        bias = self.meta_net(im_features) #(batch, ctx_dim)
        bias = bias.unsqueeze(1) #(batch, 1, ctx_dim)
        if self.ctx_beforename is not None:
            ctx_beforename_shifted = self.ctx_beforename.unsqueeze(0) + bias
        
        if self.ctx_aftername is not None:
            ctx_aftername_shifted = self.ctx_aftername.unsqueeze(0) + bias

        prompts = []
        for i in range(im_features.shape[0]):
            ctx_beforename_i = None
            if self.ctx_beforename is not None:
                ctx_beforename_i = ctx_beforename_shifted[i,:,:]
            
            ctx_aftername_i = None
            if self.ctx_aftername is not None:
                ctx_aftername_i = ctx_aftername_shifted[i,:,:]

            pts_i = self.construct_prompts(ctx_beforename_i, ctx_aftername_i)
            assert(pts_i.shape[0] == self.n_cls)
            prompts.append(pts_i)

        prompts = torch.stack(prompts)
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, return_attn_probs):
        super().__init__()
        assert(cfg.TRAINER.COCOOP.N_CTX > 0)
        ctx_init_beforename = cfg.TRAINER.COCOOP.CTX_INIT
        ctx_init_aftername = ''
        if ctx_init_beforename == '':
            ctx_init_beforename, ctx_init_aftername = None, None

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, (cfg.TRAINER.COCOOP.N_CTX, 0), (ctx_init_beforename, ctx_init_aftername))
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.return_attn_probs = return_attn_probs
        if self.return_attn_probs:
            self.text_encoder = ExposedTextEncoder(clip_model)
        else:
            self.text_encoder = TextEncoder(clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        
        logits = []
        if self.return_attn_probs:
            attn_probs = []

        for pts_i, imf_i in zip(prompts, image_features): #this is a loop across the batch
            if self.return_attn_probs:
                text_features, attn_probs_i = self.text_encoder(pts_i, tokenized_prompts)
                assert(attn_probs_i.shape == tokenized_prompts.shape) #should be (num_classes, seq_len)
            else:
                text_features = self.text_encoder(pts_i, tokenized_prompts)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
            if self.return_attn_probs:
                attn_probs.append(attn_probs_i)

        logits = torch.stack(logits)
        if self.return_attn_probs:
            attn_probs = torch.stack(attn_probs) #should be (batch_size, num_classes, seq_len)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        if self.return_attn_probs:
            return logits, attn_probs
        else:
            return logits


#@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    
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
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self, train_or_test):
        assert(train_or_test in ['train', 'test'])
        cfg = self.cfg
        classnames = {'train' : self.dm.dataset.classnames_train, 'test' : self.dm.dataset.classnames_test}[train_or_test]

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, (train_or_test == 'test' and self.record_attentropy)) #training-vs-testing-classnames problem has already been taken care of

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

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

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
        
        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
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
