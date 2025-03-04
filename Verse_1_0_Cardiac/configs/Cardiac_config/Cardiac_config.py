from detectron2.config import CfgNode as CN


def add_Cardiac_config(cfg):
    """
    Add config for Verse.
    """
    cfg.INPUT.SIZE_DIVISIBILITY = 32

    cfg.MODEL.BACKBONE.NAME = "UTNet"
    cfg.MODEL.BACKBONE.IN_CHANNELS = 3
    cfg.MODEL.BACKBONE.BASE_CHANNELS = 48
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
    cfg.MODEL.BACKBONE.REDUCE_SIZE = 8
    cfg.MODEL.BACKBONE.BLOCK_LIST = '1234'
    cfg.MODEL.BACKBONE.NUM_BLOCKS = [1, 1, 1, 1]
    cfg.MODEL.BACKBONE.PROJECTION = 'interp'
    cfg.MODEL.BACKBONE.NUM_HEADS = [4, 4, 4, 4]
    cfg.MODEL.BACKBONE.ATTN_DROP = 0.1
    cfg.MODEL.BACKBONE.PROJ_DROP = 0.1
    cfg.MODEL.BACKBONE.BOTTLENECK = False
    cfg.MODEL.BACKBONE.MAXPOOL = False
    cfg.MODEL.BACKBONE.REL_POS = True
    cfg.MODEL.BACKBONE.AUX_LOSS = False
    cfg.MODEL.PIXEL_MEAN = [0.367, 0.367, 0.367]
    cfg.MODEL.PIXEL_STD = [3.13, 3.13, 3.13]

    cfg.MODEL.MASK_FORMER = CN()
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 0.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 5.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.PRE_NORM = False
    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.INPUT.IMAGE_SIZE = 256

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8


    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 4
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.NUM_CLICK_QUERIES = 16
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "PixelFuser"
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 6
    cfg.MODEL.TRAINING_MODE = True
    cfg.MODEL.CLICK_MODEL = True
    cfg.MODEL.IMAGE_SIZE = (256, 256)
