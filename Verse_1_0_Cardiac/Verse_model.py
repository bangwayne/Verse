# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import torch
import random
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import BACKBONE_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList
from .modeling.loss import PointSampleLoss
from .utils.point_sampler import PointSampler
from .modeling.point_encoder.point_feature_map_encoder import get_batch_point_feature_map
from .modeling.backbone.utnet import UTNet
from .modeling.pixel_decoder.pixelfuser import PixelFuser
from torch.cuda.amp import GradScaler, autocast
from fvcore.nn import FlopCountAnalysis


class Verse(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            criterion: nn.Module,
            num_queries: int,
            size_divisibility: int,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            iter_num: int,
            test_iter_num: int,
            point_sample_method: str,
            training_click_mode: list[str],
            testing_click_mode: list[str]
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.iter_num = iter_num
        self.test_iter_num = test_iter_num
        self.point_sample_method = point_sample_method
        self.training_click_mode = training_click_mode
        self.testing_click_mode = testing_click_mode


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        criterion = PointSampleLoss()

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "iter_num": cfg.ITER_TRAINING.ITER_NUM,
            "test_iter_num": cfg.TEST.TEST_ITER_NUM,
            "point_sample_method": cfg.ITER_TRAINING.SAMPLE_METHOD,
            "training_click_mode": cfg.ITER_TRAINING.CLICK_MODE,
            "testing_click_mode": cfg.TEST.TEST_CLICK_MODE
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def single_inference(self, batched_inputs, outputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        """
        with torch.no_grad():
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        return mask_pred_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image['masks'].to(self.device)
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image['labels'].to(self.device),
                    "unique_labels": targets_per_image['unique_labels'].to(self.device),
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def forward(self, batched_inputs, mode="Training"):
        if mode == "Training":
            losses = self.iter_training(batched_inputs)
            return losses
        else:
            result_dict, point_dict = self.iter_inference(batched_inputs)
            return result_dict, point_dict

    def iter_training(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
            click_mode: indicate that if training or not
            epoch: indicate that epoch
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        height, width = images[0].shape[-2], images[0].shape[-1]
        # 50% chance to sample and add an element
            # Sample an element (this is just an example, modify the sampling logic as needed)
        mode = []
        sampled_element = random.choice(self.training_click_mode)  # Replace with actual logic to sample the element
        mode.append(sampled_element)
        query_index = [x["q_index"] for x in batched_inputs]
        Sampler = PointSampler()
        batch_data_with_point = Sampler.initial_test_points(batch_data=batched_inputs, device=self.device)

        epoch_num = batch_data_with_point[0]['epoch']
        print(f"epoch num: {epoch_num}")
        total_iter_num = self.iter_num

        # del batch_data_with_mask
        # stage 1: query_base seg and inter click:
        if '1' in mode:
            print("mode 1 is on")
            for iter_num in range(total_iter_num):
                for dict in batch_data_with_point:
                    dict['click_index'] = iter_num

                if iter_num == 0:
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_point).to(self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_point]
                    pos_point_tuple_list = None
                    neg_point_tuple_list = None
                else:
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point).to(self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_next_point]
                    pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point]
                    neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point]

                with autocast():
                    features = self.backbone(images.tensor)
                    outputs = self.sem_seg_head(
                        features, point_feature_batch_data, point_tuple_list,
                        pos_point_tuple_list,
                        neg_point_tuple_list,
                        click_mode='1' if iter_num != 0 else '0',
                        query_index=query_index
                    )

                    if "target" in batched_inputs[0]:
                        gt_instances = [x["target"] for x in batched_inputs]
                        targets = self.prepare_targets(gt_instances, images)
                    else:
                        targets = None

                    if iter_num == 0:
                        losses = self.criterion(outputs, targets)
                    else:
                        iter_losses = self.criterion(outputs, targets)
                    if iter_num != 0:
                        for key in iter_losses:
                            losses[key] += iter_losses[key] * 0.8

                processed_mask_results = self.single_inference(batched_inputs, outputs).sigmoid()

                for bs_index in range(len(batch_data_with_point if iter_num == 0 else batch_data_with_next_point)):
                    (batch_data_with_point if iter_num == 0 else batch_data_with_next_point)[bs_index]['seg_result'] = \
                        processed_mask_results[bs_index]

                if iter_num < total_iter_num-1:
                    if self.point_sample_method == "min_dis":
                        batch_data_with_next_point = Sampler.get_next_points(
                            batch_data_with_point if iter_num == 0 else batch_data_with_next_point,
                            device=self.device,
                            click_index=(iter_num + 1)
                        )
                    elif self.point_sample_method == "largest_component":
                        batch_data_with_next_point = Sampler.get_next_points_component(
                            batch_data_with_point if iter_num == 0 else batch_data_with_next_point,
                            device=self.device,
                            click_index=(iter_num + 1)
                        )
                # in the last training, we make it wrong to batch_data_with_point, should be with_nbe
        # stage 2: no query_base inter click:
        if '2' in mode:
            print("mode 2 is on")
            for iter_num in range(total_iter_num):
                for dict in batch_data_with_point:
                    dict['click_index'] = iter_num
                if iter_num == 0:
                    for bs_index in range(len(batch_data_with_point)):
                        batch_data_with_point[bs_index]['seg_result'] = torch.zeros(1, height, width)

                    batch_data_with_next_point_mode2 = Sampler.get_next_points_component(batch_data_with_point,
                                                                                         device=self.device,
                                                                                         click_index=(iter_num + 1))

                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point_mode2).to(
                        self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_next_point_mode2]
                    pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point_mode2]
                    neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point_mode2]

                    query_index = [x["q_index"] for x in batched_inputs]
                    with autocast():
                        features = self.backbone(images.tensor)
                        outputs = self.sem_seg_head(features, point_feature_batch_data, point_tuple_list,
                                                    pos_point_tuple_list, neg_point_tuple_list,
                                                    click_mode='2',
                                                    query_index=query_index)

                        if "target" in batched_inputs[0]:
                            gt_instances = [x["target"] for x in batched_inputs]
                            targets = self.prepare_targets(gt_instances, images)
                        # change the key here, and move the to device into the prepare_targets function
                        else:
                            targets = None

                        losses2 = self.criterion(outputs, targets)
                        if 'losses' not in locals():
                            losses = {key: torch.zeros_like(losses2[key]) for key in losses2}
                        for key in losses2:
                            losses[key] += losses2[key]
                    processed_mask_results = self.single_inference(batched_inputs, outputs)
                    processed_mask_results = processed_mask_results.sigmoid()

                    #
                    for bs_index in range(len(batch_data_with_next_point_mode2)):
                        batch_data_with_next_point_mode2[bs_index]['seg_result'] = processed_mask_results[bs_index]

                    if self.point_sample_method == "min_dis":
                        batch_data_with_next_point_mode2 = Sampler.get_next_points(batch_data_with_next_point_mode2,
                                                                                   device=self.device,
                                                                                   click_index=(iter_num + 2))
                    elif self.point_sample_method == "largest_component":
                        batch_data_with_next_point_mode2 = Sampler.get_next_points_component(
                            batch_data_with_next_point_mode2,
                            device=self.device,
                            click_index=(iter_num + 2))
                #
                else:
                    # print(f"iter_num: {iter_num}")
                    weight = 0.8
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point_mode2).to(
                        self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_next_point_mode2]
                    pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point_mode2]
                    neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point_mode2]
                    with autocast():
                        features = self.backbone(images.tensor)
                        outputs = self.sem_seg_head(features, point_feature_batch_data, point_tuple_list,
                                                    pos_point_tuple_list, neg_point_tuple_list,
                                                    click_mode='2',
                                                    query_index=query_index)
                        del point_feature_batch_data

                        if "target" in batched_inputs[0]:
                            gt_instances = [x["target"] for x in batched_inputs]
                            targets = self.prepare_targets(gt_instances, images)
                        # change the key here, and move the to device into the prepare_targets function
                        else:
                            targets = None

                        iter_losses = self.criterion(outputs, targets, cal_class_loss=False)
                        for key in iter_losses:
                            losses[key] += iter_losses[key] * weight

                    processed_mask_results = self.single_inference(batched_inputs, outputs)
                    processed_mask_results = processed_mask_results.sigmoid()
                    for bs_index in range(len(batch_data_with_next_point_mode2)):
                        batch_data_with_next_point_mode2[bs_index]['seg_result'] = processed_mask_results[bs_index]

                    if iter_num < total_iter_num - 1:
                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point_mode2 = Sampler.get_next_points(batch_data_with_next_point_mode2,
                                                                                       device=self.device,
                                                                                       click_index=(iter_num + 2))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point_mode2 = Sampler.get_next_points_component(
                                batch_data_with_next_point_mode2,
                                device=self.device,
                                click_index=(iter_num + 2))
                # in the last training, we make it wrong to batch_data_with_point, should be with_nbe
        return losses

    def iter_inference(self, batched_inputs):
        with torch.no_grad():
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            height, width = images[0].shape[-2], images[0].shape[-1]
            mode = self.testing_click_mode
            if '1' in mode:
                query_index = [int(x["q_index"]) for x in batched_inputs]
                # print(f"q_index:{query_index}")
                point_sampler = PointSampler()
                # initialize the point sampler
                batch_data_with_point = point_sampler.initial_test_points(batch_data=batched_inputs, device=self.device)
                # initialize the point list, at first, setting them to all empty [-1,-1]
                result_dict = {}
                point_dict = {}
                for iter_num in range(self.test_iter_num):
                    if iter_num == 0:
                        for batch_dict in batch_data_with_point:
                            batch_dict['click_index'] = iter_num
                        point_feature_batch_data = get_batch_point_feature_map(batch_data=batch_data_with_point).to(
                            self.device)
                        point_tuple_list = [x['points_list'] for x in batch_data_with_point]
                        features = self.backbone(images.tensor)
                        outputs = self.sem_seg_head(features, point_feature_batch_data, point_tuple_list, click_mode='0'
                                                    , query_index=query_index)
                        processed_mask_results = self.single_inference(batched_inputs, outputs)
                        processed_mask_results = processed_mask_results.sigmoid()
                        for bs_index in range(len(batch_data_with_point)):
                            batch_data_with_point[bs_index]['seg_result'] = processed_mask_results[bs_index]
                        seg_result_list = [x['seg_result'].to(self.device) for x in batch_data_with_point]
                        point_list = [x['points_list'] for x in batch_data_with_point]
                        result_dict[iter_num] = seg_result_list
                        point_dict[iter_num] = point_list

                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point = point_sampler.get_next_points(batch_data_with_point,
                                                                                       device=self.device,
                                                                                       click_index=(iter_num + 1))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point = point_sampler.get_next_points_component(
                                batch_data_with_point,
                                device=self.device,
                                click_index=(
                                        iter_num + 1))

                    else:
                        for batch_dict in batch_data_with_next_point:
                            batch_dict['click_index'] = iter_num
                        point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point).to(
                            self.device)
                        point_tuple_list = [x['points_list'] for x in batch_data_with_next_point]
                        pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point]
                        neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point]

                        features = self.backbone(images.tensor)
                        outputs = self.sem_seg_head(
                            features, point_feature_batch_data, point_tuple_list,
                            pos_point_tuple_list,
                            neg_point_tuple_list,
                            click_mode='1',
                            query_index=query_index
                        )

                        processed_results = F.interpolate(outputs["pred_masks"], size=(height, width), mode="bilinear",
                                                          align_corners=False)
                        processed_results = processed_results.sigmoid()

                        for bs_index in range(len(batch_data_with_next_point)):
                            batch_data_with_next_point[bs_index]['seg_result'] = processed_results[bs_index]

                        seg_result_list = [x['seg_result'].to(self.device) for x in batch_data_with_next_point]
                        point_list = [x['points_list'] for x in batch_data_with_next_point]
                        result_dict[iter_num] = seg_result_list
                        point_dict[iter_num] = point_list

                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point = point_sampler.get_next_points(batch_data_with_next_point,
                                                                                       device=self.device,
                                                                                       click_index=(iter_num + 1))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point = point_sampler.get_next_points_component(
                                batch_data_with_next_point,
                                device=self.device,
                                click_index=(iter_num + 1))

            if '2' in mode:
                query_index = [int(x["q_index"]) for x in batched_inputs]
                # print(f"q_index:{query_index}")
                point_sampler = PointSampler()
                # initialize the point sampler
                batch_data_with_point = point_sampler.initial_test_points(batch_data=batched_inputs, device=self.device)
                # initialize the point list, at first, setting them to all empty [-1,-1]
                result_dict = {}
                point_dict = {}
                for iter_num in range(self.test_iter_num):
                    if iter_num == 0:
                        for batch_dict in batch_data_with_point:
                            batch_dict['click_index'] = iter_num

                        for bs_index in range(len(batch_data_with_point)):
                            batch_data_with_point[bs_index]['seg_result'] = torch.zeros(1, height, width)

                        batch_data_with_next_point = point_sampler.get_next_points_component(batch_data_with_point,
                                                                                             device=self.device,
                                                                                             click_index=(iter_num + 1))
                        # print(f"this is the initial point list:{batch_data_with_next_point[bs_index]['points_list']}")
                        point_feature_batch_data = get_batch_point_feature_map(
                            batch_data=batch_data_with_next_point).to(
                            self.device)
                        point_tuple_list = [x['points_list'] for x in batch_data_with_next_point]
                        pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point]
                        neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point]
                        features = self.backbone(images.tensor)
                        outputs = self.sem_seg_head(features, point_feature_batch_data, point_tuple_list,
                                                    pos_point_tuple_list, neg_point_tuple_list,
                                                    click_mode='2',
                                                    query_index=query_index)
                        processed_mask_results = self.single_inference(batched_inputs, outputs)
                        processed_mask_results = processed_mask_results.sigmoid()
                        for bs_index in range(len(batch_data_with_next_point)):
                            batch_data_with_next_point[bs_index]['seg_result'] = processed_mask_results[bs_index]
                        seg_result_list = [x['seg_result'].to(self.device) for x in batch_data_with_next_point]
                        point_list = [x['points_list'] for x in batch_data_with_next_point]
                        result_dict[iter_num] = seg_result_list
                        point_dict[iter_num] = point_list

                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point = point_sampler.get_next_points(batch_data_with_next_point,
                                                                                       device=self.device,
                                                                                       click_index=(iter_num + 2))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point = point_sampler.get_next_points_component(
                                batch_data_with_next_point,
                                device=self.device,
                                click_index=(iter_num + 2))

                    else:
                        point_list = [x['points_list'] for x in batch_data_with_next_point]
                        # print(f"this is the iter {iter_num} point list:{point_list}")
                        for batch_dict in batch_data_with_next_point:
                            batch_dict['click_index'] = iter_num
                        point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point).to(
                            self.device)
                        point_tuple_list = [x['points_list'] for x in batch_data_with_next_point]
                        pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point]
                        neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point]
                        features = self.backbone(images.tensor)
                        outputs = self.sem_seg_head(features, point_feature_batch_data, point_tuple_list,
                                                    pos_point_tuple_list, neg_point_tuple_list,
                                                    click_mode='2',
                                                    query_index=query_index)

                        processed_results = F.interpolate(outputs["pred_masks"], size=(height, width), mode="bilinear",
                                                          align_corners=False)
                        processed_results = processed_results.sigmoid()

                        for bs_index in range(len(batch_data_with_next_point)):
                            batch_data_with_next_point[bs_index]['seg_result'] = processed_results[bs_index]

                        seg_result_list = [x['seg_result'].to(self.device) for x in batch_data_with_next_point]
                        point_list = [x['points_list'] for x in batch_data_with_next_point]
                        result_dict[iter_num] = seg_result_list
                        point_dict[iter_num] = point_list

                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point = point_sampler.get_next_points(batch_data_with_next_point,
                                                                                       device=self.device,
                                                                                       click_index=(iter_num + 2))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point = point_sampler.get_next_points_component(
                                batch_data_with_next_point,
                                device=self.device,
                                click_index=(iter_num + 2))

            return result_dict, point_dict
