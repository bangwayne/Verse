import torch
import torch.nn as nn
from .unet_utils import up_block, down_block
from .conv_trans_utils import *
import pdb
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class UTNet(Backbone):

    def __init__(self, cfg, input_shape):
        super().__init__()

        in_chan = cfg.MODEL.BACKBONE.IN_CHANNELS
        base_chan = cfg.MODEL.BACKBONE.BASE_CHANNELS
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        reduce_size = cfg.MODEL.BACKBONE.REDUCE_SIZE
        block_list = cfg.MODEL.BACKBONE.BLOCK_LIST
        num_blocks = cfg.MODEL.BACKBONE.NUM_BLOCKS
        projection = cfg.MODEL.BACKBONE.PROJECTION
        num_heads = cfg.MODEL.BACKBONE.NUM_HEADS
        attn_drop = cfg.MODEL.BACKBONE.ATTN_DROP
        proj_drop = cfg.MODEL.BACKBONE.PROJ_DROP
        bottleneck = cfg.MODEL.BACKBONE.BOTTLENECK
        maxpool = cfg.MODEL.BACKBONE.MAXPOOL
        rel_pos = cfg.MODEL.BACKBONE.REL_POS
        aux_loss = cfg.MODEL.BACKBONE.AUX_LOSS

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }

        self._out_feature_channels = {
            "res2": 48,
            "res3": 96,
            "res4": 192,
            "res5": 384,
        }

        self.aux_loss = aux_loss
        self.inc = [BasicBlock(in_chan, base_chan)]
        if '0' in block_list:
            for _ in range(1):
                self.inc.append(BasicTransBlock(base_chan, heads=num_heads[-5], dim_head=base_chan // num_heads[-5],
                                                attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                                projection=projection, rel_pos=rel_pos))
            self.up4 = up_block_trans(2 * base_chan, base_chan, num_block=0, bottleneck=bottleneck, heads=num_heads[-4],
                                      dim_head=base_chan // num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop,
                                      reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        else:
            self.inc.append(BasicBlock(base_chan, base_chan))
            self.up4 = up_block(2 * base_chan, base_chan, scale=(2, 2), num_block=2)
        self.inc = nn.Sequential(*self.inc)

        if '1' in block_list:
            self.down1 = down_block_trans(base_chan, 2 * base_chan, num_block=num_blocks[-4], bottleneck=bottleneck,
                                          maxpool=maxpool, heads=num_heads[-4], dim_head=2 * base_chan // num_heads[-4],
                                          attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                          projection=projection, rel_pos=rel_pos)
            self.up3 = up_block_trans(4 * base_chan, 2 * base_chan, num_block=0, bottleneck=bottleneck,
                                      heads=num_heads[-3], dim_head=2 * base_chan // num_heads[-3], attn_drop=attn_drop,
                                      proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                      rel_pos=rel_pos)
        else:
            self.down1 = down_block(base_chan, 2 * base_chan, (2, 2), num_block=2)
            self.up3 = up_block(4 * base_chan, 2 * base_chan, scale=(2, 2), num_block=2)

        if '2' in block_list:
            self.down2 = down_block_trans(2 * base_chan, 4 * base_chan, num_block=num_blocks[-3], bottleneck=bottleneck,
                                          maxpool=maxpool, heads=num_heads[-3], dim_head=4 * base_chan // num_heads[-3],
                                          attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                          projection=projection, rel_pos=rel_pos)
            self.up2 = up_block_trans(8 * base_chan, 4 * base_chan, num_block=0, bottleneck=bottleneck,
                                      heads=num_heads[-2], dim_head=4 * base_chan // num_heads[-2], attn_drop=attn_drop,
                                      proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                      rel_pos=rel_pos)

        else:
            self.down2 = down_block(2 * base_chan, 4 * base_chan, (2, 2), num_block=2)
            self.up2 = up_block(8 * base_chan, 4 * base_chan, scale=(2, 2), num_block=2)

        if '3' in block_list:
            self.down3 = down_block_trans(4 * base_chan, 8 * base_chan, num_block=num_blocks[-2], bottleneck=bottleneck,
                                          maxpool=maxpool, heads=num_heads[-2], dim_head=8 * base_chan // num_heads[-2],
                                          attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                          projection=projection, rel_pos=rel_pos)
            self.up1 = up_block_trans(16 * base_chan, 8 * base_chan, num_block=0, bottleneck=bottleneck,
                                      heads=num_heads[-1], dim_head=8 * base_chan // num_heads[-1], attn_drop=attn_drop,
                                      proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                      rel_pos=rel_pos)

        else:
            self.down3 = down_block(4 * base_chan, 8 * base_chan, (2, 2), num_block=2)
            self.up1 = up_block(16 * base_chan, 8 * base_chan, scale=(2, 2), num_block=2)

        if '4' in block_list:
            self.down4 = down_block_trans(8 * base_chan, 16 * base_chan, num_block=num_blocks[-1],
                                          bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-1],
                                          dim_head=16 * base_chan // num_heads[-1], attn_drop=attn_drop,
                                          proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                          rel_pos=rel_pos)
        else:
            self.down4 = down_block(8 * base_chan, 16 * base_chan, (2, 2), num_block=2)


        self.outc = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)

        if aux_loss:
            self.out1 = nn.Conv2d(8 * base_chan, num_classes, kernel_size=1, bias=True)
            self.out2 = nn.Conv2d(4 * base_chan, num_classes, kernel_size=1, bias=True)
            self.out3 = nn.Conv2d(2 * base_chan, num_classes, kernel_size=1, bias=True)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "in_chan": cfg.MODEL.BACKBONE.IN_CHANNELS,
            "base_chan": cfg.MODEL.BACKBONE.BASE_CHANNELS,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "reduce_size": cfg.MODEL.BACKBONE.REDUCE_SIZE,
            "block_list": cfg.MODEL.BACKBONE.BLOCK_LIST,
            "num_blocks": cfg.MODEL.BACKBONE.NUM_BLOCKS,
            "projection": cfg.MODEL.BACKBONE.PROJECTION,
            "num_heads": cfg.MODEL.BACKBONE.NUM_HEADS,
            "attn_drop": cfg.MODEL.BACKBONE.ATTN_DROP,
            "proj_drop": cfg.MODEL.BACKBONE.PROJ_DROP,
            "bottleneck": cfg.MODEL.BACKBONE.BOTTLENECK,
            "maxpool": cfg.MODEL.BACKBONE.MAXPOOL,
            "rel_pos": cfg.MODEL.BACKBONE.REL,
        }

    def forward(self, x):
        output_dict = {}
        x1 = self.inc(x)
        x2 = self.down1(x1)
        output_dict['res2'] = x1
        x3 = self.down2(x2)
        output_dict['res3'] = x2
        x4 = self.down3(x3)
        output_dict['res4'] = x3
        output_dict['res5'] = x4
        return output_dict

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_feature_channels
        }


