import os
import sys
import pickle
import touch
import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from build_trainer_custom import build_trainer_custom

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

#    if args.source_domains:
#        assert(False) #these should come from dataset-config file
#        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

#    if args.target_domains:
#        assert(False) #these should come from dataset-config file
#        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

#    if args.trainer:
#        assert(False) #c'mon, this should be in the damn config file!
#        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.COCOOP_ENSEMBLING = CN()
    cfg.TRAINER.COCOOP_ENSEMBLING.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP_ENSEMBLING.RANDOM_CTX_INIT = False
    cfg.TRAINER.COCOOP_ENSEMBLING.TRAIN_SEPARATELY = False
    cfg.TRAINER.COCOOP_ENSEMBLING.ONETOKEN = False
    cfg.TRAINER.COCOOP_ENSEMBLING.NOPERIOD = False

    cfg.TRAINER.COCOOP_ATTENTROPY = CN()
    cfg.TRAINER.COCOOP_ATTENTROPY.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP_ATTENTROPY.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP_ATTENTROPY.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP_ATTENTROPY.ATTENTROPY_LAMBDA = 1.0

    cfg.TRAINER.COCOOP_MULTIMODAL = CN()
    cfg.TRAINER.COCOOP_MULTIMODAL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP_MULTIMODAL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP_MULTIMODAL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP_MULTIMODAL.NUM_METANET_PASSES = 1

    cfg.TRAINER.COCOOP_EFFICIENT_ONETOKEN_ENSEMBLING = CN()
    cfg.TRAINER.COCOOP_EFFICIENT_ONETOKEN_ENSEMBLING.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP_EFFICIENT_ONETOKEN_ENSEMBLING.RANDOM_CTX_INIT = False
    cfg.TRAINER.COCOOP_EFFICIENT_ONETOKEN_ENSEMBLING.TRAIN_SEPARATELY = False
    cfg.TRAINER.COCOOP_EFFICIENT_ONETOKEN_ENSEMBLING.ONETOKEN = False
    cfg.TRAINER.COCOOP_EFFICIENT_ONETOKEN_ENSEMBLING.NOPERIOD = False

    cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED = CN()
    cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED.RANDOM_CTX_INIT = False
    cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED.TRAIN_SEPARATELY = False
    cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED.ONETOKEN = False
    cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED.NOPERIOD = False
    cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED.LAION_NUM_POSITIVES = 1
    cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED.LAION_LAMBDA = 1.0
    cfg.TRAINER.COCOOP_ENSEMBLING_LAIONIZED.GRAMLEN = 1

    #the CoOp and CoCoOp authors decided to use fp16, while the CLIP-Adapter authors decided to use fp32, so that's what we're doing for now
    cfg.TRAINER.CLIPADAPTER = CN()
    cfg.TRAINER.CLIPADAPTER.PREC = "fp32"

    #default CLIP_Adapter to use 0.2 as the authors did for ImageNet
    cfg.TRAINER.CLIPADAPTER.ALPHA = 0.2

    cfg.TRAINER.ZEROSHOTCLIP2 = CN()
    cfg.TRAINER.ZEROSHOTCLIP2.ONETOKEN = False

#    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.NAME = ""

    #these will be parsed on-the-fly as dicts of lists
    cfg.DATASET.SOURCE_DOMAINS_LIST = ''
    cfg.DATASET.TARGET_DOMAINS_LIST = ''

    #these will be parsed on-the-fly as dicts
    cfg.DATASET.FEWSHOT_FILTER_PATHS = ''
    cfg.DATASET.CLASS_SPLIT_PATHS = ''

    cfg.TRAIN.CHECKPOINT_FREQ = 1

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    assert(cfg.SEED == -1)
    if cfg.SEED >= 0:
        assert(False)
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if args.eval_only:
        train_or_test = 'test'
    else:
        assert(not args.no_train)
        train_or_test = 'train'

    laion_data_dir = (args.laion_data_dir if args.laion_data_dir != '' else None)
    trainer = build_trainer_custom(cfg, train_or_test, args.fewshot_seed, args.domain_split_index, args.class_split_type, args.eval_type, record_attentropy=args.record_attentropy, record_examples=args.record_examples, laion_data_dir=laion_data_dir)

    if args.eval_only:
        results_filename = os.path.join(cfg.OUTPUT_DIR, 'results.pkl')
        if args.skip_if_results_exists and os.path.exists(results_filename):
            print('results.pkl file already exists at "%s", skipping'%(results_filename))
            return

        if args.model_dir:
            done_filename = os.path.join(args.model_dir, 'done')
            if not os.path.exists(done_filename):
                print('Training not done yet! We were expecting a "done-file" at "%s". Please finish training.'%(done_filename))
                return

        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        _, results = trainer.test()

        #seriously, why wouldn't they already do this???
        with open(results_filename, 'wb') as f:
            pickle.dump(results, f)

        return

    if not args.no_train:
        done_filename = os.path.join(cfg.OUTPUT_DIR, 'done')
        if os.path.exists(done_filename):
            print('Already done! No need to train!')
            return

        trainer.train()

        print('Training done! Writing "done-file" to "%s"'%(done_filename))
        touch.touch(done_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory. This is where train mode saves the model, and where eval mode saves the results. So if you are training now, you would pass the same thing in as the model-dir argument when evaluating that model.")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
#    parser.add_argument(
#        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
#    )
#    parser.add_argument(
#        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
#    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
#    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--skip-if-results-exists", action="store_true", help="skip if results.pkl already exists")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    #trainer = build_trainer_custom(cfg, train_or_test, args.fewshot_seed, args.domain_split_index, args.class_split_type, args.eval_type)
    parser.add_argument(
        "--fewshot-seed", type=int, default=0, help="this maps to one of the fewshot filter files"
    )
    parser.add_argument(
        "--domain-split-index", type=int, default=0, help="this maps to a domain split"
    )
    parser.add_argument(
        "--class-split-type", type=str, default="random", help="this maps to one of the class split files"
    )
    parser.add_argument(
        "--eval-type", type=str, default="seen_domains_seen_classes", help="which domains and classes we evaluate on (only relevant for eval mode, i.e. if you pass in eval-only flag)"
    )
    parser.add_argument(
        "--laion-data-dir", type=str, default='', help='dir with LAION data for LAIONized training'
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument("--record-attentropy", action="store_true", help="record text attention map entropies (only happens during test, and only for CoCoOp and ZeroshotCLIP)")
    parser.add_argument("--record-examples", action="store_true", help="record a few examples (only happens during test, and only for CoCoOp)")
    args = parser.parse_args()
    main(args)
