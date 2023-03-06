import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset

from dassl.utils import read_image

from datasets import build_dataset_custom
from dassl.data.data_manager import build_data_loader, DatasetWrapper
from dassl.data.samplers import build_sampler
from dassl.data.transforms import INTERPOLATION_MODES, build_transform

from yacs_utils import parsedict, parsedictlist

#def __init__(self, cfg, fewshot_seed, domain_split_index, class_split_type, eval_type):

class DataManagerCustom:

    def __init__(
        self,
        cfg,
        fewshot_seed,
        domain_split_index,
        class_split_type,
        eval_type,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        self.domain_split_index = domain_split_index

        # Load dataset
        dataset = build_dataset_custom(cfg, fewshot_seed, domain_split_index, class_split_type, eval_type)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # DON'T Build train_loader_u
        train_loader_u = None
        assert(dataset.train_u is None)

        # DON'T Build val_loader
        val_loader = None
        assert(dataset.val is None)

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )


        # Attributes
        self._num_classes_train = dataset.num_classes_train
        self._num_classes_test = dataset.num_classes_test
        self._num_source_domains = len(parsedictlist(cfg.DATASET.SOURCE_DOMAINS_LIST)[str(domain_split_index)])
        self._lab2cname_train = dataset.lab2cname_train
        self._lab2cname_test = dataset.lab2cname_test
        
        # DEPRECATED Attributes
        self_num_classes, self._lab2cname = None, None

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    #DEPRECATED
    @property
    def num_classes(self):
        assert(False)
        return None
    
    @property
    def num_classes_train(self):
        return self._num_classes_train
    
    @property
    def num_classes_test(self):
        return self._num_classes_test

    @property
    def num_source_domains(self):
        return self._num_source_domains

    #DEPRECATED
    @property
    def lab2cname(self):
        assert(False)
        return None
    
    @property
    def lab2cname_train(self):
        return self._lab2cname_train
    
    @property
    def lab2cname_test(self):
        return self._lab2cname_test

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = parsedictlist(cfg.DATASET.SOURCE_DOMAINS_LIST)[str(self.domain_split_index)]
        target_domains = parsedictlist(cfg.DATASET.TARGET_DOMAINS_LIST)[str(self.domain_split_index)]

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes_train", f"{self.num_classes_train:,}"])
        table.append(["# classes_test", f"{self.num_classes_test:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))
