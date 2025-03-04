from detectron2.config import CfgNode as CN


def add_training_config(cfg):
    """
    Add training config for Verse.
    """
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.DATALOADER.NUM_WORKERS = 3
    cfg.DATASETS.PATH = "/research/cbim/vast/bg654/Desktop/jupyproject/imask2former/datasets"
    cfg.DATASETS.NAME = "ACDC"
    cfg.DATASETS.NAME_LIST = ["ACDC", "MM", "MnM2", "MYO_C0", "MYO_T2", "MYO_LGE", "LASCARQS"]
    # If you train in the combined dataset, you should focus on the NAMELIST
    cfg.DATASETS.ITER_MODEL = True
    cfg.DATASETS.TEST_MODEL = "train"

    cfg.TRAINING = CN()
    cfg.TRAINING.DIST = True
    cfg.TRAINING.NODE_RANK = 0
    cfg.TRAINING.INIT_METHOD = "env://"
    cfg.TRAINING.BUCKET_CAP_MD = 25
    cfg.TRAINING.WORK_DIR = "/research/cbim/vast/bg654/Desktop/jupyproject/Verse_git/exp"
    cfg.TRAINING.NUM_NODES = 1
    cfg.TRAINING.NUM_EPOCH = 75
    cfg.TRAINING.RESUME = None
    cfg.TRAINING.BATCH_SIZE = 8

    # solver config
    # weight decay on embedding
    cfg.ITER_TRAINING = CN()
    cfg.ITER_TRAINING.ITER_NUM = 3
    cfg.ITER_TRAINING.SAMPLE_METHOD = "largest_component"
    cfg.ITER_TRAINING.CLICK_MODE = ['1', '2']

    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.5
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0

    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.WARMUP_EPOCH = 10
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # cfg.TEST = CN()
    cfg.TEST.TEST_ITER_NUM = 20
    cfg.TEST.DIST = True
    cfg.TEST.DRAW = False
    cfg.TEST.TEST_CLICK_MODE = ['2']
    # cfg.TEST.TEST_ITER_NUM = 3


