from detectron2.config import CfgNode as CN


def add_testing_config(cfg):

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATASETS.PATH = "/research/cbim/vast/bg654/Desktop/jupyproject/imask2former/datasets"
    cfg.DATASETS.NAME = "ACDC"
    cfg.DATASETS.TEST_MODEL = "test"
    cfg.DATASETS.ITER_MODEL = True

    cfg.TRAINING = CN()
    cfg.TRAINING.DIST = True
    cfg.TRAINING.NODE_RANK = 0
    cfg.TRAINING.NUM_NODES = 1
    cfg.TRAINING.INIT_METHOD = "env://"
    cfg.TRAINING.BUCKET_CAP_MD = 25
    cfg.TRAINING.WORK_DIR = "imask2former/exp/work_dir"
    cfg.TRAINING.RESUME = None
    cfg.TRAINING.BATCH_SIZE = 16

    cfg.ITER_TRAINING = CN()
    cfg.ITER_TRAINING.ITER_NUM = 3
    cfg.ITER_TRAINING.SAMPLE_METHOD = "largest_component"
    cfg.ITER_TRAINING.CLICK_MODE = ['2']

    cfg.TEST.DIST = True
    cfg.TEST.DRAW = False
    cfg.TEST.TEST_ITER_NUM = 20
    cfg.TEST.TEST_CLICK_MODE = ['2']
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.5
    # cfg.MODEL.MA