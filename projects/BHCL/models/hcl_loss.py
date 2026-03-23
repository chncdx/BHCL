import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmrotate.registry import MODELS


def gather(tensor: torch.Tensor,
           keep_grad: bool = True):
    if dist.is_initialized():
        num_GPUs = dist.get_world_size()
        # gathered tensors must have the same shape except dimension 0
        shape_0_list = [None for _ in range(num_GPUs)]
        dist.all_gather_object(shape_0_list, tensor.shape[0])
        if tensor.dim() > 1:
            shape_other = list(tensor.shape)[1:]
        else:
            shape_other = []
        tensor_list = [torch.empty([shape_0_list[i]] + shape_other, dtype=tensor.dtype, device=tensor.device)
                       for i in range(num_GPUs)]
        dist.all_gather(tensor_list, tensor)
        if keep_grad:
            GPU_ID = dist.get_rank()
            tensor_list[GPU_ID] = tensor
        return torch.concat(tensor_list, dim=0)
    else:
        return tensor


def supervised_contrastive_loss(features_1: torch.Tensor,
                                labels_1: torch.Tensor,
                                features_2: torch.Tensor,
                                labels_2: torch.Tensor,
                                self_mask: torch.Tensor,
                                temperature: float):
    logits = features_1 @ features_2.T / temperature
    max_logits, _ = logits.max(dim=1, keepdim=True)
    logits = logits - max_logits.detach()
    exp_logits = logits.exp() * self_mask
    log_probs = logits - exp_logits.sum(dim=1, keepdim=True).log()
    pos_mask = (labels_1.unsqueeze(1) == labels_2.unsqueeze(0)) * self_mask
    mean_log_probs = - (log_probs * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
    return mean_log_probs


def balanced_supervised_contrastive_loss(features_1: torch.Tensor,
                                         labels_1: torch.Tensor,
                                         features_2: torch.Tensor,
                                         labels_2: torch.Tensor,
                                         self_mask: torch.Tensor,
                                         temperature: float):
    logits = features_1 @ features_2.T / temperature
    max_logits, _ = logits.max(dim=1, keepdim=True)
    logits = logits - max_logits.detach()
    pos_mask = (labels_1.unsqueeze(1) == labels_2.unsqueeze(0)) * self_mask
    cls_count = torch.bincount(labels_2)
    exp_logits = logits.exp() * self_mask
    exp_logits = exp_logits / cls_count[labels_2]
    log_probs = logits - exp_logits.sum(dim=1, keepdim=True).log()
    mean_log_probs = - (log_probs * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
    return mean_log_probs


@MODELS.register_module()
class HierarchicalContrastiveLoss(nn.Module):

    def __init__(self,
                 num_classes: int,
                 num_levels: int,
                 hierarchical_labels: List[Tuple[str]],
                 begin_level: int = 0,
                 proj_head_input_dim: int = 256,
                 proj_head_hidden_dim: int = 512,
                 proj_head_output_dim: int = 128,
                 temperature: float = 0.1,
                 level_weights: List[float] = None,
                 loss_weight: float = 1.0):
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = num_levels

        self.hierarchical_labels = - torch.ones((num_classes, num_levels), dtype=torch.int64)
        cls_map = {}
        cls_id = 0
        for hierarchical_label in hierarchical_labels:
            cls_map[hierarchical_label[-1]] = cls_id
            cls_id += 1
        for i, hierarchical_label in enumerate(hierarchical_labels):
            for j, cls in enumerate(hierarchical_label):
                self.hierarchical_labels[i, j] = cls_map[cls]

        self.current_epoch = 1
        self.begin_level = begin_level
        self.proj_head = nn.Sequential(
            nn.Linear(proj_head_input_dim, proj_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_head_hidden_dim, proj_head_output_dim)
        )
        self.temperature = temperature
        if level_weights is None:
            self.level_weights = [math.exp(1 / (num_levels - begin_level - l)) for l in range(0, num_levels - begin_level)]
        else:
            self.level_weights = level_weights
        self.loss_weight = loss_weight

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self,
                queries: torch.Tensor,
                labels: torch.Tensor):
        queries = self.proj_head(queries)
        queries = F.normalize(queries, p=2, dim=1)
        queries = gather(queries)
        labels = gather(labels)
        loss_dict = self.forward_standard(queries, labels)
        for loss in loss_dict.keys():
            loss_dict[loss] = loss_dict[loss] * self.loss_weight
        return loss_dict

    def forward_standard(self,
                         queries: torch.Tensor,
                         labels: torch.Tensor):
        loss_dict = {}
        hierarchical_labels = self.hierarchical_labels.to(labels.device)
        for l in range(self.begin_level, self.num_levels):
            labels_l = hierarchical_labels[labels, l]
            indices = torch.nonzero((labels_l != -1) & ((labels_l.unsqueeze(1) == labels_l.unsqueeze(0)).sum(dim=1) > 1)).squeeze(1)
            if len(indices) == 0:
                break
            if l == self.begin_level:
                N = len(indices)
            queries_l = queries[indices]
            labels_l = labels_l[indices]
            self_mask = 1 - torch.eye(len(indices), dtype=queries.dtype, device=queries.device)
            loss = supervised_contrastive_loss(queries_l, labels_l, queries_l, labels_l, self_mask, self.temperature)
            loss_dict[f'lv{l}_loss'] = loss.sum() / N * self.level_weights[l - self.begin_level] / sum(self.level_weights)
        if loss_dict:
            return loss_dict
        else:
            return { f'lv{self.begin_level}_loss' : queries.sum() * 0.0 }


class PrototypeBank_EMA(nn.Module):

    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 num_levels: int,
                 epsilon: float = 0.1):
        super().__init__()
        prototypes = torch.zeros((num_classes, feature_dim), dtype=torch.float32)
        prototype_labels = torch.arange(0, num_classes, 1, dtype=torch.int64)
        self.register_buffer('prototypes', prototypes)
        self.register_buffer('prototype_labels', prototype_labels)
        self.num_levels = num_levels
        self.epsilon = epsilon
    
    def getPrototypes(self, device):
        return self.prototypes.to(device), self.prototype_labels.to(device)
    
    def updatePrototypes(self, queries, labels, level):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            with torch.no_grad():
                update_values = torch.zeros_like(self.prototypes)
                update_values.index_add_(0, labels, queries)
                cls_cnt = torch.zeros_like(self.prototype_labels, dtype=torch.float32)
                cls_cnt.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
                update_values /= cls_cnt.clamp_min(1.0).unsqueeze(1)
                unique_labels = labels.unique()
                momentum = 1 - self.epsilon ** (self.num_levels - level)
                self.prototypes[unique_labels] = momentum * self.prototypes[unique_labels] + \
                    (1 - momentum) * update_values[unique_labels]
                self.prototypes[unique_labels] = F.normalize(self.prototypes[unique_labels], p=2, dim=1)
        if world_size > 1:
            dist.broadcast(self.prototypes, src=0)


@MODELS.register_module()
class BalancedHierarchicalContrastiveLoss(HierarchicalContrastiveLoss):

    def __init__(self, *args, **kwargs):
        proj_head_output_dim = kwargs.get('proj_head_output_dim', 128)
        epsilon = kwargs.pop('epsilon', 0.1)
        self.warmup_epoch = kwargs.pop('warmup_epoch', 5)
        super().__init__(*args, **kwargs)
        self.prototype_bank = PrototypeBank_EMA(self.num_classes, proj_head_output_dim, self.num_levels, epsilon)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self,
                queries: torch.Tensor,
                labels: torch.Tensor):
        queries = self.proj_head(queries)
        queries = F.normalize(queries, p=2, dim=1)
        queries = gather(queries)
        labels = gather(labels)
        if self.current_epoch <= self.warmup_epoch:
            loss_dict = self.forward_standard(queries, labels)
        else:
            loss_dict = self.forward_balanced(queries, labels)
        for loss in loss_dict.keys():
            loss_dict[loss] = loss_dict[loss] * self.loss_weight
        return loss_dict

    def forward_standard(self,
                         queries: torch.Tensor,
                         labels: torch.Tensor):
        loss_dict = {}
        hierarchical_labels = self.hierarchical_labels.to(labels.device)
        for l in range(self.begin_level, self.num_levels):
            labels_l = hierarchical_labels[labels, l]
            indices = torch.nonzero((labels_l != -1) & ((labels_l.unsqueeze(1) == labels_l.unsqueeze(0)).sum(dim=1) > 1)).squeeze(1)
            if len(indices) == 0:
                break
            if l == self.begin_level:
                N = len(indices)
            queries_l = queries[indices]
            labels_l = labels_l[indices]
            self_mask = 1 - torch.eye(len(indices), dtype=queries.dtype, device=queries.device)
            loss = supervised_contrastive_loss(queries_l, labels_l, queries_l, labels_l, self_mask, self.temperature)
            loss_dict[f'lv{l}_loss'] = loss.sum() / N * self.level_weights[l - self.begin_level] / sum(self.level_weights)
            self.prototype_bank.updatePrototypes(queries_l, labels_l, l)
        if loss_dict:
            return loss_dict
        else:
            return { f'lv{self.begin_level}_loss' : queries.sum() * 0.0 }

    def forward_balanced(self,
                         queries: torch.Tensor,
                         labels: torch.Tensor):
        loss_dict = {}
        hierarchical_labels = self.hierarchical_labels.to(labels.device)
        prototypes, prototype_labels = self.prototype_bank.getPrototypes(queries.device)
        for l in range(self.begin_level, self.num_levels):
            labels_l = hierarchical_labels[labels, l]
            prototype_labels_l = hierarchical_labels[prototype_labels, l]
            indices = torch.nonzero(labels_l != -1).squeeze(1)
            prototype_indices = torch.nonzero(prototype_labels_l != -1).squeeze(1)
            if len(indices) == 0:
                break
            if l == self.begin_level:
                N = len(indices)
            queries_l = queries[indices]
            prototypes_l = prototypes[prototype_indices]
            labels_l = labels_l[indices]
            prototype_labels_l = prototype_labels_l[prototype_indices]
            self_mask = torch.concat([
                1 - torch.eye(len(indices), dtype=queries.dtype, device=queries.device),
                torch.ones((len(indices), len(prototype_indices)), dtype=queries.dtype, device=queries.device)
            ], dim=1)
            loss = balanced_supervised_contrastive_loss(queries_l, labels_l,
                                                        torch.concat([queries_l, prototypes_l], dim=0),
                                                        torch.concat([labels_l, prototype_labels_l], dim=0),
                                                        self_mask, self.temperature)
            loss_dict[f'lv{l}_loss'] = loss.sum() / N * self.level_weights[l - self.begin_level] / sum(self.level_weights)
            self.prototype_bank.updatePrototypes(queries_l, labels_l, l)
        if loss_dict:
            return loss_dict
        else:
            return { f'lv{self.begin_level}_loss' : queries.sum() * 0.0 }
