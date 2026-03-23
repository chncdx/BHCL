import glob
import logging
import os.path as osp
from typing import List

from mmengine.logging import print_log

from mmrotate.registry import DATASETS
from mmrotate.datasets.dota import DOTADataset


@DATASETS.register_module()
class ShipRSImageNetDataset(DOTADataset):

    METAINFO = {
        'classes': (
            # Level 0
            'Ship',
            # Level 1
            'Warship',
            'Merchant',
            # Level 2
            'Submarine',
            'Aircraft Carrier',
            'Destroyer',
            'Ticonderoga',
            'Frigate',
            'Patrol',
            'Landing',
            'Commander',
            'Auxiliary Ship',
            'Container Ship',
            'RoRo',
            'Cargo',
            'Barge',
            'Tugboat',
            'Ferry',
            'Yacht',
            'Sailboat',
            'Fishing Vessel',
            'Oil Tanker',
            'Hovercraft',
            'Motorboat',
            # Level 3
            'Enterprise',
            'Nimitz',
            'Midway',
            'Atago DD',
            'Arleigh Burke DD',
            'Hatsuyuki DD',
            'Hyuga DD',
            'Asagiri DD',
            'Perry FF',
            'YuTing LL',
            'YuDeng LL',
            'YuDao LL',
            'YuZhao LL',
            'Austin LL',
            'Osumi LL',
            'Wasp LL',
            'LSD 41 LL',
            'LHA LL',
            'Medical Ship',
            'Test Ship',
            'Training Ship',
            'AOE',
            'Masyuu AS',
            'Sanantonio AS',
            'EPF'
        )
    }

    def __init__(self, **kwargs):
        super().__init__(img_suffix='bmp', **kwargs)

    def load_data_list(self) -> List[dict]:
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in '
                                 f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)

                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.split()
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]
                        # make sure the correct process of category names with space
                        cls_name = ' '.join(bbox_info[8:-1])
                        # relabel instances from other* categories
                        if cls_name.startswith('Other '):
                            cls_name = cls_name[6:]
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[-1])
                        if difficulty > self.diff_thr:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list

    def __getitems__(self, possibly_batched_index: list[int]) -> list[dict]:
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = []
            for idx in possibly_batched_index:
                tmp = self.prepare_data(idx)
                if tmp is None:
                    raise Exception('Test time pipline should not get `None` '
                                    'data_sample')
                data.append(tmp)
            return data

        data = []
        for idx in possibly_batched_index:
            flag = False
            for _ in range(self.max_refetch + 1):
                tmp1 = self.prepare_data(idx)
                if tmp1 is None:
                    idx = self._rand_another()
                    continue
                tmp2 = self.prepare_data(idx)
                data.extend([tmp1, tmp2])
                flag = True
                break
            if not flag:
                raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                                'Please check your image path and pipeline')
        return data
