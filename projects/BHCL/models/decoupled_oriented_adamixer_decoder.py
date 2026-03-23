
import sys
from datetime import datetime

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist

from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import ConfigType, InstanceList
from mmdet.models.utils.misc import empty_instances, unpack_gt_instances
from mmrotate.registry import MODELS

from .oriented_adamixer_decoder import OrientedAdaMixerDecoder

@MODELS.register_module()
class DecoupledOrientedAdaMixerDecoder(OrientedAdaMixerDecoder):

    def __init__(self,
                 leaf_indices: list[int] = None,
                 decouple: bool = False,
                 hcl_loss: ConfigType = None,
                 *args, **kwargs):
        num_stages = kwargs.get('num_stages', None)
        super().__init__(*args, **kwargs)
        self.leaf_indices = leaf_indices
        self.decouple = decouple
        if hcl_loss is not None:
            self.hcl_loss = nn.ModuleList([
                MODELS.build(hcl_loss) for _ in range(num_stages)
            ])
        else:
            self.hcl_loss = None

    def loss(self, x: tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs
        for item in batch_gt_instances:
            item.bboxes = get_box_tensor(item.bboxes)

        query_content = torch.cat(
            [res.pop('query_content')[None, ...] for res in rpn_results_list])   # bs, num_query, 256
        results_list = rpn_results_list
        losses = {}
        for stage in range(self.num_stages):
            stage_loss_weight = self.stage_loss_weights[stage]

            # bbox head forward and loss
            bbox_results = self.bbox_loss(
                stage=stage,
                x=x,
                query_content=query_content,
                results_list=results_list,
                batch_img_metas=batch_img_metas,
                batch_gt_instances=batch_gt_instances)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            query_content = bbox_results['query_content']
            results_list = bbox_results['results_list']

            # calculate balanced hierarchical contrastive loss
            if self.hcl_loss is not None:
                sampling_results = bbox_results['sampling_results']
                queries = []
                labels = []
                # only use predictions which are matched to foreground objects
                for i in range(len(sampling_results)):
                    queries.append(query_content[i, sampling_results[i].pos_inds, :])
                    labels.append(sampling_results[i].pos_gt_labels)
                
                queries = torch.cat(queries, dim=0)
                if self.decouple:
                    queries, _ = queries.chunk(2, dim=-1)
                labels = torch.cat(labels, dim=0)

                loss_dict = self.hcl_loss[stage](queries, labels)
                
                for loss in loss_dict.keys():
                    losses[f's{stage}.{loss}'] = loss_dict[loss]

        return losses

    def predict_bbox(self,
                     x: tuple[Tensor],
                     batch_img_metas: list[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        xyzrt_list = [res.query_xyzrt for res in rpn_results_list]
        query_xyzrt = torch.stack(xyzrt_list)  # bs, num_query, 5

        query_content = torch.cat(
            [res.pop('query_content')[None, ...] for res in rpn_results_list])   # bs, num_query, 256
        if all([xyzrt.shape[0] == 0 for xyzrt in xyzrt_list]):
            # There is no proposal in the whole batch
            return empty_instances(
                batch_img_metas, x[0].device, task_type='bbox')

        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage, x, query_xyzrt, query_content,
                                              batch_img_metas)
            query_content = bbox_results['query_content']
            cls_score = bbox_results['cls_score']
            bboxes_list = bbox_results['detached_bboxes_list']
            query_xyzrt = bbox_results['query_xyzrt']

        num_classes = self.bbox_head[-1].num_classes

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        topk_inds_list = []
        results_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score_per_img = cls_score[img_id]
            # if leaf_indices provided, use it to create a mask matrix
            # to avoid making non_leaf predictions during inference.
            if self.leaf_indices is not None:
                mask_matrix = torch.zeros_like(cls_score_per_img)
                mask_matrix[:, self.leaf_indices] = 1
                cls_score_per_img = cls_score_per_img * mask_matrix
            scores_per_img, topk_inds = cls_score_per_img.flatten(0, 1).topk(
                self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_inds % num_classes
            bboxes_per_img = bboxes_list[img_id][topk_inds // num_classes]
            topk_inds_list.append(topk_inds)
            if rescale and bboxes_per_img.size(0) > 0:
                assert batch_img_metas[img_id].get('scale_factor') is not None
                scale_factor = bboxes_per_img.new_tensor(
                    batch_img_metas[img_id]['scale_factor']).repeat((1, 2))
                # Notice: Due to keep ratio when resize in data preparation,
                # the angle(radian) will not rescale.
                radian_factor = scale_factor.new_ones((scale_factor.size(0), 1))
                scale_factor = torch.cat([scale_factor, radian_factor], dim=-1)
                bboxes_per_img = (
                    bboxes_per_img.view(bboxes_per_img.size(0), -1, 5) /
                    scale_factor).view(bboxes_per_img.size()[0], -1)

            results = InstanceData()
            results.bboxes = bboxes_per_img
            results.scores = scores_per_img
            results.labels = labels_per_img
            results_list.append(results)
        return results_list
