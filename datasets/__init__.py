from datasets.domainnet_custom import DomainNetCustom

def build_dataset_custom(cfg, fewshot_seed, domain_split_index, class_split_type, eval_type):
    if cfg.DATASET.NAME == 'DomainNetCustom':
        return DomainNetCustom(cfg, fewshot_seed, domain_split_index, class_split_type, eval_type)
    else:
        assert(False)
