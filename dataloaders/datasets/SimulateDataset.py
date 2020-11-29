#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/27 9:30
# @Author  : XuPenglei
# @Site    : 
# @File    : SimulateDataset.py
# @Software: PyCharm
# @email  : xupenglei87@163.com
# @Description: 模拟遥感数据集

from __future__ import print_function, division
import os
from skimage.io import imread
import numpy as np
from torch.utils.data import Dataset
from fnmatch import fnmatch
from PIL import Image
from torchvision import transforms
from dataloaders import custom_transforms as tr

class SimulateRemoteSensing(Dataset):
    """
    模拟遥感数据集，一个图片比较大
    """
    NUM_CLASSES = 2

    def __init__(self,
                 base_dir,
                 X_sub_dir,
                 Y_sub_dir,
                 patch_size,
                 SR,
                 to_train
                 ):
        super().__init__()
        self._image_dir = os.path.join(base_dir,X_sub_dir)
        self._cat_dir = os.path.join(base_dir, Y_sub_dir)

        images = [n for n in os.listdir(self._image_dir) if fnmatch(n,'*.tif') or fnmatch(n,'*.tiff')]

        self.image_filenames = [os.path.join(self._image_dir,n) for n in images]
        self.label_filenames = [os.path.join(self._cat_dir,n) for n in images]

        img = Image.open(self.image_filenames[0])
        self.image_height = img.size[0]
        self.image_width = img.size[1]
        self.num_bands = len(img.split())
        self.patch_size = patch_size

        self.SR = SR
        self.to_train = to_train

    @property
    def patch_rows_per_img(self):
        return int((self.image_height - self.patch_size) /
                   self.patch_size) + 1

    @property
    def patch_cols_per_img(self):
        return int((self.image_width - self.patch_size) /
                   self.patch_size) + 1

    @property
    def patches_per_img(self):
        return self.patch_rows_per_img * self.patch_cols_per_img

    @property
    def num_imgs(self):
        return len(self.image_filenames)

    @property
    def num_patches(self):
        return self.patches_per_img * self.num_imgs

    def _get_patch(self, filenames, patch_idx, SR=1):
        img_idx = int(patch_idx / self.patches_per_img)
        img_patch_idx = patch_idx % self.patches_per_img
        row_idx = int(img_patch_idx / self.patch_cols_per_img)
        col_idx = img_patch_idx % self.patch_cols_per_img
        img = Image.open(filenames[img_idx])
        bbox = (col_idx*self.patch_size*SR,row_idx*self.patch_size*SR,
                (col_idx+1)*self.patch_size*SR,(row_idx+1)*self.patch_size*SR)
        patch_image = img.crop(bbox)
        return patch_image

    def __len__(self):
        return self.num_patches

    def train_tansform(self,sample):
        composed_transoforms = transforms.Compose([
            transforms.RandomChoice([
                tr.LosslessRotate(p=0.5),
                tr.RandomVerticalFlip(p=0.5),
                tr.RandomHorizontalFlip(p=0.5),
                tr.RandomTranspose45(p=0.5),
                tr.RandomTranspose235(p=0.5)
            ]),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transoforms(sample)

    def val_tansform(self,sample):
        composed_transoforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transoforms(sample)

    def __getitem__(self, index):
        _img = self._get_patch(self.image_filenames,index,SR=1)
        _label = self._get_patch(self.label_filenames,index, SR=self.SR)
        sample = {'image': _img, 'label': _label}
        if self.to_train:
            return self.train_tansform(sample)
        else:
            return self.val_tansform(sample)


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt


    voc_train = SimulateRemoteSensing(
        base_dir=r'F:\Data\MassachusettsBuilding\mass_buildings\train',
        X_sub_dir='satLR',
        Y_sub_dir='map',
        patch_size=128,
        SR=4,
        to_train=True
    )

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)





