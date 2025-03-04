import logging
import fvcore.nn.weight_init as weight_init
import math
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Tuple
import copy
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .SimpleMaskDecoder import SimpleMaskDecoder, CAResBlock, MultiScaleDownsamplingModule
from ..point_encoder.point_feature_map_encoder import get_resize_feature_map, get_point_feature
from ..point_encoder.point_encoder import PromptEncoder


def deepcopy_tensor_list(tensor_list):
    return [tensor.clone() for tensor in tensor_list]


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class PixelFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv = CAResBlock(dim, dim)

    def forward(self, pixel_flat: torch.Tensor, feature_size) -> torch.Tensor:
        h, w = feature_size
        hw, bs, embed_dim = pixel_flat.shape

        # make sure that the input has the proper size
        assert hw == h * w, f"Expected {h * w}, but got {hw}"
        # adjust the size map
        pixel_flat = pixel_flat.view(h, w, bs, self.dim)
        pixel_flat = pixel_flat.permute(2, 3, 0, 1).contiguous()  # [Batch_size, dim, H, W]

        x = self.conv(pixel_flat)

        # return to the original map
        x = x.permute(2, 3, 0, 1).contiguous()  # [H, W, Batch_size, dim]
        x = x.view(hw, bs, self.dim)  # [(H*W), Batch_size, dim]

        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        return self.upsample(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(DownsampleBlock, self).__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        return self.downsample(x)


def create_block_list(in_channels, out_channels):
    block_list = nn.ModuleList()
    block_list.append(UpsampleBlock(in_channels, out_channels, scale_factor=2))
    block_list.append(UpsampleBlock(out_channels, out_channels, scale_factor=2))
    block_list.append(DownsampleBlock(out_channels, out_channels, scale_factor=4))
    block_list.append(UpsampleBlock(out_channels, out_channels, scale_factor=2))
    block_list.append(UpsampleBlock(out_channels, out_channels, scale_factor=2))
    return block_list


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):
    _version = 2

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            num_click_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            enforce_input_project: bool,
            training_mode: bool,
            image_size: tuple,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = False
        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.point_embedding_dim = hidden_dim
        self.num_layers = dec_layers
        #######
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.pos_transformer_self_attention_layers = nn.ModuleList()
        self.pos_transformer_cross_attention_layers = nn.ModuleList()
        self.pos_transformer_ffn_layers = nn.ModuleList()

        self.neg_transformer_self_attention_layers = nn.ModuleList()
        self.neg_transformer_cross_attention_layers = nn.ModuleList()
        self.neg_transformer_ffn_layers = nn.ModuleList()

        self.transformer_feature_query_cross_attention_layers = nn.ModuleList()
        self.transformer_pixel_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.pos_transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.neg_transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.pos_transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.neg_transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.pos_transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.neg_transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            ###################################################
            self.transformer_feature_query_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_pixel_ffn_layers.append(
                PixelFFN(dim=hidden_dim)
            )
        #########################
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.single_object_query = num_queries
        self.num_queries = num_classes * num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_classes * num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_classes * num_queries, hidden_dim)

        self.num_classes = num_classes
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.aggregate_class_layer = MLP(num_queries, num_queries, 1, num_layers=2)

        self.simple_decoders = nn.ModuleList()
        for i in range(self.num_feature_levels):
            self.simple_decoders.append(SimpleMaskDecoder(in_channels=hidden_dim, num_res_blocks=2, out_channels=32))

        self.sample_layer_list = create_block_list(hidden_dim, hidden_dim)
        self.res_query_layer_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(dec_layers - 1)])

        # click mode
        self.point_encoder = MultiScaleDownsamplingModule(in_channels=3, out_channels=hidden_dim)
        self.training_mode = training_mode
        self.image_size = image_size
        
    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["num_click_queries"] = cfg.MODEL.MASK_FORMER.NUM_CLICK_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        # click_mode or not
        ret["training_mode"] = cfg.MODEL.TRAINING_MODE
        ret["image_size"] = cfg.MODEL.IMAGE_SIZE

        return ret

    def forward(self, x, point_feature_mask, point_tuple_list=None, pos_point_tuple_list=None,
                neg_point_tuple_list=None, click_mode='0', query_index=[]):
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        # print(f'point_tuple_list:{point_tuple_list}')
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            # remain the H*W
            pos.append(self.pe_layer(x[i], None).flatten(2))
            # flatten NxCxHxW to NxCxHW
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        if click_mode == '0':
            object_query_output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            object_query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            # Q*C --> QxBxC
            pixel_feature_list = deepcopy_tensor_list(src)
            # now pixel_feature_list = [tensor[B,C,H,W], ...., ....]
            query_output = self.extract_query_vector(object_query_output, label_index_list=query_index)
            query_embed = self.extract_query_vector(object_query_embed, label_index_list=query_index)
            single_object_query_num = query_output.shape[0]
            output_masks_all_layer = self.process_object_query_layer(
                query_output, query_embed,
                pixel_feature_list=pixel_feature_list, size_list=size_list, pos_list=pos, attn_mask=None,
                query_num=single_object_query_num)

            predictions_mask = []
            for layer_index in range(self.num_layers):
                predictions_mask.append(output_masks_all_layer[layer_index])  # concat to [16, num_objects, H, W]
            assert len(predictions_mask) == self.num_layers
            out = {
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(predictions_mask
                                                  )
            }

        elif click_mode == '1':
            all_object_query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            all_object_query_output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            object_query_output = self.extract_query_vector(all_object_query_output, label_index_list=query_index)
            object_query_embed = self.extract_query_vector(all_object_query_embed, label_index_list=query_index)
            single_object_query_num = object_query_output.shape[0]

            (sparse_pos_click_query_embed, sparse_pos_click_query_output, sparse_neg_click_query_embed,
             sparse_neg_click_query_output) = self.get_point_queries(
                pos_point_tuple_list, neg_point_tuple_list, input_image_size=self.image_size
            )
            pixel_feature_list = deepcopy_tensor_list(src)
            pos_click_query_output_list, neg_click_query_output_list = self.dense_click_query_embed(
                pixel_feature_list, pos_point_tuple_list,
                neg_point_tuple_list)

            update_point_query_tuple = (pos_point_tuple_list, neg_point_tuple_list)

            pos_query_embed = sparse_pos_click_query_embed.repeat(2, 1, 1)
            neg_query_embed = sparse_neg_click_query_embed.repeat(2, 1, 1)
            click_query_embed = (pos_query_embed, neg_query_embed)
            click_query_output = (sparse_pos_click_query_output, pos_click_query_output_list,
                                  sparse_neg_click_query_output, neg_click_query_output_list)

            output_masks_all_layer = self.process_click_and_object_layer(
                object_query_output, object_query_embed, click_query_output, click_query_embed,
                pixel_feature_list, size_list, pos, point_feature_mask=point_feature_mask,
                query_num=single_object_query_num, update_point_query_tuple=update_point_query_tuple)

            predictions_mask = []

            for layer_index in range(self.num_layers):
                predictions_mask.append(output_masks_all_layer[layer_index])

            assert len(predictions_mask) == self.num_layers

            out = {
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(predictions_mask
                                                  )
            }

        elif click_mode == '2':

            (sparse_pos_click_query_embed, sparse_pos_click_query_output, sparse_neg_click_query_embed,
             sparse_neg_click_query_output) = self.get_point_queries(
                pos_point_tuple_list, neg_point_tuple_list, input_image_size=self.image_size
            )
            pixel_feature_list = deepcopy_tensor_list(src)
            # now pixel_feature_list = [tensor[HW,B,C], ...., ....]
            click_query_num = sparse_pos_click_query_embed.shape[0]

            pos_click_query_output_list, neg_click_query_output_list = self.dense_click_query_embed(
                pixel_feature_list, pos_point_tuple_list,
                neg_point_tuple_list)

            update_point_query_tuple = (pos_point_tuple_list,
                                        neg_point_tuple_list)

            pos_query_embed = sparse_pos_click_query_embed.repeat(2, 1, 1)
            neg_query_embed = sparse_neg_click_query_embed.repeat(2, 1, 1)
            click_query_embed = (pos_query_embed, neg_query_embed)
            click_query_output = (sparse_pos_click_query_output, pos_click_query_output_list,
                                  sparse_neg_click_query_output, neg_click_query_output_list)

            output_masks_all_layer = self.process_pos_neg_click_single_object_layer(
                click_query_output, click_query_embed,
                pixel_feature_list, size_list, pos, attn_mask=None,
                point_feature_mask=point_feature_mask, point_tuple_list=point_tuple_list, update_point_query_tuple=
                update_point_query_tuple)

            predictions_mask = []

            for layer_index in range(self.num_layers):
                predictions_mask.append(output_masks_all_layer[layer_index])

            assert len(predictions_mask) == self.num_layers
            out = {
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(predictions_mask,
                                                  )
            }

        return out

    def extract_query_vector(self, query_vector, label_index_list):
        object_num = self.num_classes
        batch_size = query_vector.shape[1]
        Q = query_vector.shape[0] // object_num
        query_list = []
        for i in range(batch_size):
            label_index = label_index_list[i]
            index = label_index - 1
            start = index * Q
            end = (index + 1) * Q
            extracted_vector = query_vector[start:end, i, :].unsqueeze(1)
            query_list.append(extracted_vector)
        extracted_query_vector = torch.cat(query_list, dim=1)
        # print(f"extract_query_vector_shape: {extracted_query_vector.shape}")
        return extracted_query_vector

    def process_object_query_layer(self, object_query_output, object_query_embed, pixel_feature_list,
                                   size_list, pos_list, attn_mask=None, query_num=4):
        output_masks_all_layer = []
        num_of_level = len(size_list)
        attn_mask = attn_mask
        res_pixel_feature = None  # Will be initialized in the first iteration

        for i in range(self.num_layers):
            level_index = i % num_of_level
            # print(f'level_index{level_index}')
            attn_masks_level_index = (level_index + 1) % 3
            # print(i)
            if isinstance(attn_mask, torch.Tensor):
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            original_pixel_feature = pixel_feature_list[level_index]
            pixel_feature = original_pixel_feature

            # Now the pixel_feature is the size (HW, B, C)
            pixel_feature = self.add_resized_pixel_feature(pixel_feature, res_pixel_feature, iteration=i)
            # Now the pixel_feature is the size (HW, B, C)

            pos = pos_list[level_index]
            # print(f"object_query_shape:{object_query_output.shape}")
            object_query_output = self.transformer_cross_attention_layers[i](
                object_query_output, pixel_feature,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos, query_pos=object_query_embed
            )

            object_query_output = self.transformer_self_attention_layers[i](
                object_query_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=object_query_embed
            )

            object_query_output = self.transformer_ffn_layers[i](object_query_output)
            # the click part

            pixel_feature = self.transformer_feature_query_cross_attention_layers[i](
                pixel_feature, object_query_output,
                pos=object_query_embed, query_pos=pos
            )

            pixel_feature = self.transformer_pixel_ffn_layers[i](pixel_feature, size_list[level_index])
            # (HW,B,C)
            pixel_feature_list[level_index] = pixel_feature
            pixel_feature_size = size_list[level_index]
            res_pixel_feature = pixel_feature.clone()
            single_outputs_mask = self.single_forward_prediction_heads(pixel_feature, pixel_feature_size, level_index)
            attn_mask = F.interpolate(single_outputs_mask, size=size_list[attn_masks_level_index], mode="bilinear",
                                      align_corners=False)
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, query_num,
                                                                            1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()
            output_masks_all_layer.append(single_outputs_mask)

        return output_masks_all_layer

    def process_click_and_object_layer(self, object_query_output, object_query_embed,
                                       click_query_output, click_query_embed, pixel_feature_list,
                                       size_list, pos_list, point_feature_mask=None,
                                       query_num=4, update_point_query_tuple=None):
        output_masks_all_layer = []
        num_of_level = len(size_list)

        sparse_pos_click_query_output, pos_click_query_output_list, sparse_neg_click_query_output, neg_click_query_output_list = \
            click_query_output[0], click_query_output[1], click_query_output[2], click_query_output[3]
        pos_query_embed, neg_query_embed = click_query_embed[0], click_query_embed[1]

        point_feature_mask = point_feature_mask.squeeze(1)
        prev_mask = point_feature_mask[:, 2, :, :]  # [B, 256, 256]
        attn_mask = F.interpolate(prev_mask.unsqueeze(1), size=size_list[0], mode="bilinear",
                                  align_corners=False)
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, query_num,
                                                                        1).flatten(0, 1) < 0.5).bool()
        # (B*h, query_num, size)
        res_pixel_feature = None
        res_pos_query_output = None
        res_neg_query_output = None
        # Will be initialized in the first iteration
        for i in range(self.num_layers):
            level_index = i % num_of_level
            attn_masks_level_index = (level_index + 1) % 3

            original_pos_query_output = pos_click_query_output_list[level_index]
            # (Q,B,C)
            original_neg_query_output = neg_click_query_output_list[level_index]

            pos_query_num = original_pos_query_output.shape[0]
            # Q
            neg_query_num = original_neg_query_output.shape[0]

            pos_click_attn_mask = attn_mask[:, 0, :].unsqueeze(1).repeat(1, 2 * pos_query_num, 1)
            # (B*H,2Q,HW)
            neg_click_attn_mask = ~(attn_mask[:, 0, :].unsqueeze(1).repeat(1, 2 * neg_query_num, 1))

            if isinstance(attn_mask, torch.Tensor):
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            if isinstance(pos_click_attn_mask, torch.Tensor):
                pos_click_attn_mask[torch.where(pos_click_attn_mask.sum(-1) == pos_click_attn_mask.shape[-1])] = False
            if isinstance(neg_click_attn_mask, torch.Tensor):
                neg_click_attn_mask[torch.where(neg_click_attn_mask.sum(-1) == neg_click_attn_mask.shape[-1])] = False
            original_pixel_feature = pixel_feature_list[level_index]
            pixel_feature = original_pixel_feature
            # Now the pixel_feature is the size (HW, B, C)
            pixel_feature = self.add_resized_pixel_feature(pixel_feature, res_pixel_feature, iteration=i)
            # Now the pixel_feature is the size (HW, B, C)

            pos = pos_list[level_index]
            # print(f"object_query_shape:{object_query_output.shape}")
            object_query_output = self.transformer_cross_attention_layers[i](
                object_query_output, pixel_feature,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos, query_pos=object_query_embed
            )
            object_query_output = self.transformer_self_attention_layers[i](
                object_query_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=object_query_embed
            )
            object_query_output = self.transformer_ffn_layers[i](object_query_output)
            # the click part
            pos_feature_query_output = self.add_res_query_output(original_pos_query_output, res_pos_query_output,
                                                                 iteration=i)
            neg_feature_query_output = self.add_res_query_output(original_neg_query_output, res_neg_query_output,
                                                                 iteration=i)
            pos_query_output = torch.cat([sparse_pos_click_query_output, pos_feature_query_output], dim=0)
            neg_query_output = torch.cat([sparse_neg_click_query_output, neg_feature_query_output], dim=0)
            pos_click_query_output = self.pos_transformer_cross_attention_layers[i](
                pos_query_output, pixel_feature,
                memory_mask=pos_click_attn_mask,
                memory_key_padding_mask=None,
                pos=pos, query_pos=pos_query_embed
            )
            neg_click_query_output = self.neg_transformer_cross_attention_layers[i](
                neg_query_output, pixel_feature,
                memory_mask=neg_click_attn_mask,
                memory_key_padding_mask=None,
                pos=pos, query_pos=neg_query_embed
            )

            # Self Attention
            pos_click_query_output = self.pos_transformer_self_attention_layers[i](
                pos_click_query_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=pos_query_embed
            )
            neg_click_query_output = self.neg_transformer_self_attention_layers[i](
                neg_click_query_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=neg_query_embed
            )

            pos_click_query_output = self.pos_transformer_ffn_layers[i](pos_click_query_output)
            neg_click_query_output = self.neg_transformer_ffn_layers[i](neg_click_query_output)

            res_pos_query_output = pos_click_query_output[pos_query_num:, :, :].clone()
            res_neg_query_output = neg_click_query_output[neg_query_num:, :, :].clone()

            all_query_output = torch.concat((object_query_output, pos_click_query_output, neg_click_query_output),
                                            dim=0)
            all_query_embed = torch.concat((object_query_embed, pos_query_embed, neg_query_embed), dim=0)

            pixel_feature = self.transformer_feature_query_cross_attention_layers[i](
                pixel_feature, all_query_output,
                pos=all_query_embed, query_pos=pos
            )

            pixel_feature = self.transformer_pixel_ffn_layers[i](pixel_feature, size_list[level_index])
            pixel_feature_list[level_index] = pixel_feature

            pixel_feature_size = size_list[level_index]
            res_pixel_feature = pixel_feature.clone()

            single_outputs_mask = self.single_forward_prediction_heads(pixel_feature, pixel_feature_size, level_index)
            attn_mask = F.interpolate(single_outputs_mask, size=size_list[attn_masks_level_index], mode="bilinear",
                                      align_corners=False)
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, query_num,
                                                                            1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            if level_index == 2:
                (pos_point_tuple_list, neg_point_tuple_list) = update_point_query_tuple
                pos_click_query_output_list, neg_click_query_output_list = self.dense_click_query_embed(
                    pixel_feature_list, pos_point_tuple_list,
                    neg_point_tuple_list)
            output_masks_all_layer.append(single_outputs_mask)

        return output_masks_all_layer

    def process_pos_neg_click_single_object_layer(self, click_query_output, click_query_embed, pixel_feature_list,
                                                  size_list, pos_list, attn_mask=None, point_feature_mask=None,
                                                  point_tuple_list=None,
                                                  update_point_query_tuple=None):

        output_masks_all_layer = []
        num_of_level = len(size_list)
        sparse_pos_click_query_output, pos_click_query_output_list, sparse_neg_click_query_output, neg_click_query_output_list = \
            click_query_output[0], click_query_output[1], click_query_output[2], click_query_output[3]
        pos_query_embed, neg_query_embed = click_query_embed[0], click_query_embed[1]

        attn_mask = attn_mask
        res_pixel_feature = None  # Will be initialized in the first iteration
        res_pos_query_output = None
        res_neg_query_output = None
        for i in range(self.num_layers):
            level_index = i % num_of_level
            attn_masks_level_index = (level_index + 1) % 3

            original_pixel_feature = pixel_feature_list[level_index]
            original_pos_query_output = pos_click_query_output_list[level_index]
            original_neg_query_output = neg_click_query_output_list[level_index]
            pos_query_num = original_pos_query_output.shape[0]
            neg_query_num = original_neg_query_output.shape[0]

            if isinstance(attn_mask, torch.Tensor):
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            if attn_mask is not None:
                pos_attn_mask = attn_mask[:, :2*pos_query_num, :]
                neg_attn_mask = attn_mask[:, 2*pos_query_num:, :]
            else:
                pos_attn_mask = None
                neg_attn_mask = None

            pixel_feature = original_pixel_feature
            # Now the pixel_feature is the size (HW, B, C)
            pos = pos_list[level_index]
            pixel_feature = self.add_resized_pixel_feature(pixel_feature, res_pixel_feature, iteration=i)
            pos_feature_query_output = self.add_res_query_output(original_pos_query_output, res_pos_query_output, iteration=i)
            neg_feature_query_output = self.add_res_query_output(original_neg_query_output, res_neg_query_output, iteration=i)


            pos_query_output = torch.cat([sparse_pos_click_query_output, pos_feature_query_output], dim=0)
            neg_query_output = torch.cat([sparse_neg_click_query_output, neg_feature_query_output], dim=0)

            pos_click_query_output = self.pos_transformer_cross_attention_layers[i](
                pos_query_output, pixel_feature,
                memory_mask=pos_attn_mask,
                memory_key_padding_mask=None,
                pos=pos, query_pos=pos_query_embed
            )
            neg_click_query_output = self.neg_transformer_cross_attention_layers[i](
                neg_query_output, pixel_feature,
                memory_mask=neg_attn_mask,
                memory_key_padding_mask=None,
                pos=pos, query_pos=neg_query_embed
            )

            # Self Attention
            pos_click_query_output = self.pos_transformer_self_attention_layers[i](
                pos_click_query_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=pos_query_embed
            )
            neg_click_query_output = self.neg_transformer_self_attention_layers[i](
                neg_click_query_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=neg_query_embed
            )

            pos_click_query_output = self.pos_transformer_ffn_layers[i](pos_click_query_output)
            neg_click_query_output = self.neg_transformer_ffn_layers[i](neg_click_query_output)

            res_pos_query_output = pos_click_query_output[pos_query_num:, :, :].clone()
            res_neg_query_output = neg_click_query_output[neg_query_num:, :, :].clone()

            all_click_query_output = torch.concat((pos_click_query_output, neg_click_query_output), dim=0)
            all_click_query_embed = torch.concat((pos_query_embed, neg_query_embed), dim=0)

            pixel_feature = self.transformer_feature_query_cross_attention_layers[i](
                pixel_feature, all_click_query_output,
                pos=all_click_query_embed, query_pos=pos
            )

            pixel_feature = self.transformer_pixel_ffn_layers[i](pixel_feature, size_list[level_index])

            pixel_feature_list[level_index] = pixel_feature

            pixel_feature_size = size_list[level_index]
            res_pixel_feature = pixel_feature.clone()

            single_outputs_mask = self.single_forward_prediction_heads(pixel_feature, pixel_feature_size, level_index)

            binary_attn_mask = (single_outputs_mask.sigmoid() > 0.5).float()
            resize_scale = 256 / size_list[attn_masks_level_index][0]

            point_feature_mask = get_resize_feature_map(point_tuple_list, resize_scale=resize_scale,
                                                        target_size=size_list[attn_masks_level_index],
                                                        mask=binary_attn_mask.squeeze(1))
            point_feature_mask = point_feature_mask.squeeze(1)

            attn_mask = self.generate_new_attn_mask(point_feature_mask,
                                                    pos_query_num=2*pos_query_num, neg_query_num=2*neg_query_num)
            del point_feature_mask

            if level_index == 2:
                (pos_point_tuple_list, neg_point_tuple_list) = update_point_query_tuple
                pos_click_query_output_list, neg_click_query_output_list = self.dense_click_query_embed(
                    pixel_feature_list, pos_point_tuple_list,
                    neg_point_tuple_list)

            output_masks_all_layer.append(single_outputs_mask)

        return output_masks_all_layer

    def single_forward_prediction_heads(self, pixel_features, pixel_features_size,
                                        level_index):
        pixel_features = self.process_feature_dim(pixel_features, pixel_features_size)
        simple_decoder = self.simple_decoders[level_index]
        outputs_mask = simple_decoder(pixel_features)
        return outputs_mask

    def generate_new_attn_mask(self, point_feature, pos_query_num=8, neg_query_num=8):
        existing_mask = point_feature[:, 2, :, :]  # shape: (B, H, W)
        new_attn_mask = existing_mask.clamp(min=0, max=1)
        attn_mask = new_attn_mask.unsqueeze(1)
        #
        pos_attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, pos_query_num,
                                                                  1).flatten(0, 1) < 0.5).bool()
        neg_attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, neg_query_num,
                                                                  1).flatten(0, 1) > 0.5).bool()
        attn_mask = torch.concat((pos_attn_mask, neg_attn_mask), dim=1)

        attn_mask = attn_mask.detach()

        return attn_mask

    def add_resized_pixel_feature(self,
                                  pixel_feature: torch.Tensor,
                                  res_pixel_feature: Optional[torch.Tensor],
                                  iteration: int,
                                  ) -> torch.Tensor:
        """
        Adds a resized residual pixel feature from the previous iteration to the current pixel feature.

        Arguments:
            pixel_feature (torch.Tensor): Current pixel feature with shape (H*W, B, C).
            res_pixel_feature (Optional[torch.Tensor]): Residual pixel feature from the previous iteration.
            iteration (int): Current iteration index.

        Returns:
            torch.Tensor: Updated pixel feature with the resized residual added.
        """
        # Initialize `res_pixel_feature` if it is None
        if res_pixel_feature is None:
            res_pixel_feature = torch.zeros_like(pixel_feature)

        # Only resize and add residual feature after the first iteration
        if iteration > 0:
            # Determine the dimensions for reshaping
            feature_height = feature_width = math.isqrt(res_pixel_feature.shape[0])
            bs, channel = pixel_feature.shape[1], pixel_feature.shape[2]

            # Select the appropriate sampling layer
            sample_layer = self.sample_layer_list[iteration - 1]
            # Reshape and permute residual for resizing
            res_pixel_feature_reshaped = res_pixel_feature.view(feature_height, feature_width, bs, channel).permute(2,
                                                                                                                    3,
                                                                                                                    0,
                                                                                                                    1)
            # Resize using the sampling layer
            resized_res_pixel_feature = sample_layer(res_pixel_feature_reshaped)
            # Flatten and permute back to match the shape of `pixel_feature`
            resized_res_pixel_feature = resized_res_pixel_feature.flatten(2).permute(2, 0, 1)
            # Add resized residual feature to the current pixel feature
            pixel_feature = pixel_feature + resized_res_pixel_feature

        return pixel_feature

    def add_res_query_output(self,
                             query_input: torch.Tensor,
                             res_query_output: torch.Tensor,
                             iteration: int,
                             ) -> torch.Tensor:
        """
        Adds a resized residual pixel feature from the previous iteration to the current pixel feature.
        Arguments:
            query_input (torch.Tensor): Current pixel feature with shape (Q, B, C).
            res_query_output (Optional[torch.Tensor]): Residual pixel feature from the previous iteration.
            iteration (int): Current iteration index.
        Returns:
            torch.Tensor: Updated pixel feature with the resized residual added.
        """
        # Initialize `res_pixel_feature` if it is None
        if res_query_output is None:
            add_res_query_output = torch.zeros_like(query_input)
        # Only resize and add residual feature after the first iteration
        if iteration > 0:
            sample_layer = self.res_query_layer_list[iteration - 1]
            add_res_query_output = sample_layer(res_query_output)
        query_output = query_input + add_res_query_output

        return query_output

    def get_point_queries(self,
                          pos_point_tuple_list: List[List[Tuple[torch.Tensor, torch.Tensor]]],
                          neg_point_tuple_list: List[List[Tuple[torch.Tensor, torch.Tensor]]],
                          input_image_size: Tuple[int, int],
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes positive and negative points into query embeddings.

        Arguments:
            pos_point_tuple_list (List[Tuple[torch.Tensor, torch.Tensor]]): List of positive point tuples.
            neg_point_tuple_list (List[Tuple[torch.Tensor, torch.Tensor]]): List of negative point tuples.
            input_image_size (Tuple[int, int]): The input image size for normalization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            click_query_embed, click_query_output, neg_click_query_embed, neg_click_query_output
            Each of shape (Q, B, C), where Q is the number of points, B is the batch size, and C is the embedding dimension.
        """
        # Stack positive and negative points and labels into batches
        batch_pos_points = torch.stack([item[0][0] for item in pos_point_tuple_list], dim=0)  # [B, N, 2]
        batch_pos_labels = torch.stack([item[0][1] for item in pos_point_tuple_list], dim=0)  # [B, N, 1]
        batch_neg_points = torch.stack([item[0][0] for item in neg_point_tuple_list], dim=0)  # [B, N, 2]
        batch_neg_labels = torch.stack([item[0][1] for item in neg_point_tuple_list], dim=0)  # [B, N, 1]

        device = batch_neg_points.device
        embed_dim = self.point_embedding_dim
        # Initialize the PromptEncoder
        point_encoder = PromptEncoder(
            embed_dim=embed_dim,
            input_image_size=input_image_size,
        ).to(device)

        # Embed positive and negative points
        pos_point_query = point_encoder(batch_pos_points, batch_pos_labels, pad=False)
        neg_point_query = point_encoder(batch_neg_points, batch_neg_labels, pad=False)

        # Permute to match desired shape (Q, B, C) for both positive and negative embeddings
        click_query_embed = pos_point_query.permute(1, 0, 2)
        click_query_output = click_query_embed.clone()  # Optionally separate if needed

        neg_click_query_embed = neg_point_query.permute(1, 0, 2)
        neg_click_query_output = neg_click_query_embed.clone()  # Optionally separate if needed

        return click_query_embed, click_query_output, neg_click_query_embed, neg_click_query_output

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_masks": a}
                for a in zip(outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

    @staticmethod
    def dense_click_query_embed(
            pixel_feature_list,
            pos_point_tuple_list: List[List[Tuple[torch.Tensor, torch.Tensor]]],
            neg_point_tuple_list: List[List[Tuple[torch.Tensor, torch.Tensor]]]
    ):
        rescale_list = [8, 4, 2]
        pos_click_query_list = []
        neg_click_query_list = []
        for level_index in range(len(pixel_feature_list)):
            original_pixel_feature = pixel_feature_list[level_index]

            dense_pos_point_query_output = get_point_feature(original_pixel_feature, pos_point_tuple_list,
                                                             rescale_list[level_index])
            dense_neg_point_query_output = get_point_feature(original_pixel_feature, neg_point_tuple_list,
                                                             rescale_list[level_index])
            pos_point_query_output = dense_pos_point_query_output
            neg_point_query_output = dense_neg_point_query_output
            pos_click_query_list.append(pos_point_query_output)
            neg_click_query_list.append(neg_point_query_output)
            del dense_pos_point_query_output, dense_neg_point_query_output, pos_point_query_output, neg_point_query_output

        return pos_click_query_list, neg_click_query_list

    @staticmethod
    def process_feature_dim(pixel_feature, pixel_features_size):
        max_hw, batch_size, embed_dim = pixel_feature.shape
        max_height, max_width = pixel_features_size
        assert max_hw == max_height * max_width, "The hw dimension must be equal to height * width"
        # Reshape the tensor to (batch_size, embed_dim, h, w)
        max_pixel_features = pixel_feature.view(max_height, max_width, batch_size, embed_dim)
        processed_features = max_pixel_features.permute(2, 3, 0, 1)
        return processed_features
