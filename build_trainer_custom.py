from trainers.cocoop import CoCoOp
from trainers.clip_adapter import CLIP_Adapter
from trainers.zsclip import ZeroshotCLIP, ZeroshotCLIP2
from trainers.coop import CoOp
from trainers.cocoop_ensembling import CoCoOpEnsembling
from trainers.cocoop_attentropy import CoCoOpAttentropy

def build_trainer_custom(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type, record_attentropy=False):
    if cfg.TRAINER.NAME == 'CoCoOp':
        return CoCoOp(cfg, train_or_test, fewshot_seed, domain_split_index, class_split_type, eval_type, record_attentropy=record_attentropy)
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
    else:
        assert(False)
