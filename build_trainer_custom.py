from trainers.cocoop import CoCoOp
from trainers.clip_adapter import CLIP_Adapter
from trainers.zsclip import ZeroshotCLIP, ZeroshotCLIP2
from trainers.coop import CoOp
from trainers.cocoop_ensembling import CoCoOpEnsembling
from trainers.cocoop_attentropy import CoCoOpAttentropy
from trainers.cocoop_multimodal import CoCoOpMultimodal
from trainers.cocoop_efficient_onetoken_ensembling import CoCoOpEfficientOnetokenEnsembling
from trainers.cocoop_ensembling_laionized import CoCoOpEnsemblingLAIONized
from trainers.cocoop_ensembling_laionized_fixed_classname import CoCoOpEnsemblingLAIONizedFixedClassname

def build_trainer_custom(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type, record_attentropy=False, record_examples=False, laion_data_dir=None):
    if cfg.TRAINER.NAME == 'CoCoOp':
        return CoCoOp(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type, record_attentropy=record_attentropy, record_examples=record_examples)
    elif cfg.TRAINER.NAME == 'CLIP_Adapter':
        return CLIP_Adapter(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type)
    elif cfg.TRAINER.NAME == 'ZeroshotCLIP':
        return ZeroshotCLIP(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type, record_attentropy=record_attentropy)
    elif cfg.TRAINER.NAME == 'ZeroshotCLIP2':
        return ZeroshotCLIP2(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type)
    elif cfg.TRAINER.NAME == 'CoOp':
        return CoOp(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type)
    elif cfg.TRAINER.NAME == 'CoCoOpEnsembling':
        return CoCoOpEnsembling(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type)
    elif cfg.TRAINER.NAME == 'CoCoOpAttentropy':
        return CoCoOpAttentropy(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type, record_attentropy=record_attentropy)
    elif cfg.TRAINER.NAME == 'CoCoOpMultimodal':
        return CoCoOpMultimodal(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type, record_attentropy=record_attentropy)
    elif cfg.TRAINER.NAME == 'CoCoOpEfficientOnetokenEnsembling':
        return CoCoOpEfficientOnetokenEnsembling(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type)
    elif cfg.TRAINER.NAME == 'CoCoOpEnsemblingLAIONized':
        return CoCoOpEnsemblingLAIONized(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type, laion_data_dir)
    elif cfg.TRAINER.NAME == 'CoCoOpEnsemblingLAIONizedFixedClassname':
        return CoCoOpEnsemblingLAIONizedFixedClassname(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type, laion_data_dir)
    else:
        assert(False)
