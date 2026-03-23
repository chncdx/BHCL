
# (CVPR 2026) Balanced Hierarchical Contrastive Learning with Decoupled Queries for Fine-grained Object Detection in Remote Sensing Images

## Abstract

Fine-grained remote sensing datasets often use hierarchical label structures to differentiate objects in a coarse-to-fine manner, with each object annotated across multiple levels. However, embedding this semantic hierarchy into the representation learning space to improve fine-grained detection performance remains challenging. Previous studies have applied supervised contrastive learning at different hierarchical levels to group objects under the same parent class while distinguishing sibling subcategories. Nevertheless, they overlook two critical issues: (1) imbalanced data distribution across the label hierarchy causes high-frequency classes to dominate the learning process, and (2) learning semantic relationships among categories interferes with class-agnostic localization. To address these issues, we propose a balanced hierarchical contrastive loss combined with a decoupled learning strategy within the detection transformer (DETR) framework. The proposed loss introduces learnable class prototypes and equilibrates gradients contributed by different classes at each hierarchical level, ensuring that each hierarchical class contributes equally to the loss computation in every mini-batch. The decoupled strategy separates DETR's object queries into classification and localization sets, enabling task-specific feature extraction and optimization. Experiments on three fine-grained datasets with hierarchical annotations demonstrate that our method outperforms state-of-the-art approaches.

## Environment Setup

Our code is implemented based on the [MMRotate](https://github.com/open-mmlab/mmrotate/tree/1.x) library.

Here, we provide the commands to set up the [conda](https://repo.anaconda.com/miniconda/) environment required for the experiments:

```shell
# 1. adjust the pip version (we have found that higher versions of pip may cause bugs when using openmim and mmrotate)
pip install pip==25.2
# 2. create a new conda environment & activate
conda create -n bhcl python=3.9 -y
conda activate bhcl
# 3. install pytorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# 4. install mmengine and mmcv
pip install openmim==0.3.9
mim install "mmengine==0.10.7"
mim install "mmcv==2.2.0"
mim install "mmdet==3.3.0"
# 5. install this repository in editable mode
pip install -v -e .
# 6. manually downgrade the automatically installed numpy version to 1.x
pip install numpy==1.26.4
```

Find `mmdet/__init__.py` in your conda environment (likely at `/root/miniconda3/envs/bhcl/lib/python3.9/site-packages/mmdet/__init__.py`) and comment out the lines that check the mmcv version (at approximately lines 16–19):

```python
# assert (mmcv_version >= digit_version(mmcv_minimum_version)
#         and mmcv_version < digit_version(mmcv_maximum_version)), \
#     f'MMCV=={mmcv.__version__} is used but incompatible. ' \
#     f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'
```

Run the demo provided by MMRotate to check the environment setup:

```shell
# 1. download the checkpoint file
mim download mmrotate --config oriented-rcnn-le90_r50_fpn_1x_dota --dest demo
# 2. run the demo
python demo/image_demo.py demo/demo.jpg demo/oriented-rcnn-le90_r50_fpn_1x_dota.py demo/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file demo/result.jpg
```

The environment setup is verified if `demo/result.jpg` is produced with perfect detection results.

## Dataset Preparation

- [ShipRSImageNet-v1.0](https://github.com/zzndream/ShipRSImageNet)
- [FAIR1M-v1.0/v2.0](https://www.aircas.ac.cn/dtxw/kydt/202409/t20240918_7364598.html)

Our code requires converting the dataset annotations into DOTA format. For the ShipRSImageNet-v1.0 dataset, the 'Dock' category needs to be removed. And images in the FAIR1M-v1.0/v2.0 datasets should be cropped into $1024 \times 1024$ patches with a 200-pixel overlap. We lost the preprocessing Python script during data migration last year and will re-provide them soon.

We noticed that the dataset provided in the official ShipRSImageNet repository has been updated to v1.1, and the v1.0 version is no longer available for download. Therefore, we provide the preprocessed V1.0 dataset directly here: [ShipRSImageNet-V1-DOTA_format.zip](https://pan.baidu.com/s/1qaNVzGLW_rc3swZC6gPuBQ) (Baidu Drive; password: `bhcl`).

The dataset config files need to be modified according to the storage location of the datasets. For example (`projects/BHCL/shiprs.py`):

```python
dataset_type = 'ShipRSImageNetDataset'
data_root = '/opt/data/private/data/ShipRSImageNet_V1_DOTA_format/'  # modify here
backend_args = None
# ... ...
```

## Training

Run the following command for **single-GPU training**:

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

For example:

```shell
python tools/train.py projects/BHCL/configs/shiprs_bhcl_decoupled_6x.py
```

Run the following command for **multi-GPU training**:

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

For example:

```shell
./tools/dist_train.sh projects/BHCL/configs/shiprs_bhcl_decoupled_6x.py 4
```

## Testing

Run the following command for **single-GPU testing**:

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

For example:

```shell
python tools/test.py projects/BHCL/configs/shiprs_bhcl_decoupled_6x.py work_dirs/shiprs_bhcl_decoupled_6x/best_dota_*mAP_epoch_66.pth
```

Run the following command for **multi-GPU testing**:

```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
```

For example:

```shell
./tools/dist_test.sh projects/BHCL/configs/shiprs_bhcl_decoupled_6x.py work_dirs/shiprs_bhcl_decoupled_6x/best_dota_*mAP_epoch_66.pth 4
```

## Citation

If you find our work useful for your research, please cite our paper using the following BibTeX entry:

```bibtex
@inproceedings{chen2026bhcl,
  title={Balanced Hierarchical Contrastive Learning with Decoupled Queries for Fine-grained Object Detection in Remote Sensing Images},
  author={Chen, Jingzhou and Chen, Dexin and Xiong, Fengchao and Qian, Yuntao and Xiao, Liang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026},
}
```
