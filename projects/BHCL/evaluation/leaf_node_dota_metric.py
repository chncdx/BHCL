
import tempfile
import os.path as osp
from typing import List
from collections import OrderedDict
from multiprocessing import get_context

import numpy as np
import torch

from mmengine.logging import MMLogger
from mmdet.evaluation.functional import average_precision
from mmrotate.evaluation import eval_rbbox_map
from mmrotate.registry import METRICS
from mmrotate.evaluation.metrics import DOTAMetric
from mmrotate.evaluation.functional.mean_ap import get_cls_results, tpfp_default, print_map_summary


def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   box_type='rbox',
                   dataset=None,
                   logger=None,
                   nproc=4,
                   leaf_indices=None):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple], optional): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Defaults to None.
        iou_thr (float): IoU threshold to be considered as matched.
            Defaults to 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        box_type (str): Box type. If the QuadriBoxes is used, you need to
            specify 'qbox'. Defaults to 'rbox'.
        dataset (list[str] | str, optional): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Defaults to None.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details.
            Defaults to None.
        nproc (int): Processes used for computing TP and FP.
            Defaults to 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i, box_type)

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [box_type for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                if box_type == 'rbox':
                    gt_areas = bbox[:, 2] * bbox[:, 3]
                elif box_type == 'qbox':
                    pts = bbox.reshape(*bbox.shape[:-1], 4, 2)
                    roll_pts = torch.roll(pts, 1, dims=-2)
                    xyxy = torch.sum(
                        pts[..., 0] * roll_pts[..., 1] -
                        roll_pts[..., 0] * pts[..., 1],
                        dim=-1)
                    gt_areas = 0.5 * torch.abs(xyxy)
                else:
                    raise NotImplementedError
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        aps_excluding_others = []
        for i, cls_result in enumerate(eval_results):
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
                if leaf_indices is not None and i in leaf_indices:
                    aps_excluding_others.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0
        if leaf_indices is not None:
            mean_ap_excluding_others = np.array(aps_excluding_others).mean().item() if aps else 0.0
        else:
            mean_ap_excluding_others = None

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, mean_ap_excluding_others, eval_results


@METRICS.register_module()
class LeafNodeDOTAMetric(DOTAMetric):

    def __init__(self,
                 leaf_indices: List[int] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.leaf_indices = leaf_indices


    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        if self.merge_patches:
            # convert predictions to txt format and dump to zip file
            zip_path = self.merge_results(preds, outfile_prefix)
            logger.info(f'The submission file save at {zip_path}')
            return eval_results
        else:
            # convert predictions to coco format and dump to json file
            _ = self.results2json(preds, outfile_prefix)
            if self.format_only:
                logger.info('results are saved in '
                            f'{osp.dirname(outfile_prefix)}')
                return eval_results

        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']
            dets = [pred['pred_bbox_scores'] for pred in preds]

            mean_aps = []
            mean_aps_excluding_others = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, mean_ap_excluding_others, _ = eval_rbbox_map(
                    dets,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=self.use_07_metric,
                    box_type=self.predict_box_type,
                    dataset=dataset_name,
                    logger=logger,
                    leaf_indices=self.leaf_indices)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                if self.leaf_indices is not None:
                    mean_aps_excluding_others.append(mean_ap_excluding_others)
                    eval_results[f'*AP{int(iou_thr * 100):02d}'] = round(mean_ap_excluding_others, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            if self.leaf_indices is not None:
                eval_results['*mAP'] = sum(mean_aps_excluding_others) / len(mean_aps_excluding_others)
                eval_results.move_to_end('*mAP', last=False)
            eval_results.move_to_end('mAP', last=False)
        else:
            raise NotImplementedError
        return eval_results