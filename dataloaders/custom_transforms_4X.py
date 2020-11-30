import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

from torchvision import transforms

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        imgLR = sample['imageLR']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        imgLR = np.array(imgLR).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        imgLR /= 255.0
        imgLR -= self.mean
        imgLR /= self.std

        return {'image': img,
                'imageLR':imgLR,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        imgLR = sample['imageLR']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        imgLR = np.array(imgLR).astype(np.float32).transpose((2,0,1))
        mask = np.array(mask).astype(np.float32)/255.0

        img = torch.from_numpy(img).float()
        imgLR = torch.from_numpy(imgLR).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'imageLR':imgLR,
                'label': mask}


class RandomHorizontalFlip(object):
    def __init__(self,p=0.5):
        self.p=p
    def __call__(self, sample):
        img = sample['image']
        imgLR = sample['imageLR']
        mask = sample['label']
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            imgLR = imgLR.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'imageLR':imgLR,
                'label': mask}

class RandomVerticalFlip(object):
    def __init__(self,p=0.5):
        self.p=p
    def __call__(self, sample):
        img = sample['image']
        imgLR = sample['imageLR']
        mask = sample['label']
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            imgLR = imgLR.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': img,
                'imageLR':imgLR,
                'label': mask}

class RandomTranspose45(object):
    """45度对角线翻转"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img = sample['image']
        imgLR = sample['imageLR']
        mask = sample['label']
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM).rotate(270)
            imgLR = imgLR.transpose(Image.FLIP_TOP_BOTTOM).rotate(270)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM).rotate(270,Image.NEAREST)
        return {'image': img,
                'imageLR':imgLR,
                'label': mask}

class RandomTranspose235(object):
    """45度对角线翻转"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img = sample['image']
        imgLR = sample['imageLR']
        mask = sample['label']
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270)
            imgLR = imgLR.transpose(Image.FLIP_LEFT_RIGHT).rotate(270)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT).rotate(270,Image.NEAREST)
        return {'image': img,
                'imageLR':imgLR,
                'label': mask}

class LosslessRotate(object):
    def __init__(self, p=0.5):
        self.degree = [0,90,180,270]
        self.p = p

    def __call__(self, sample):
        img = sample['image']
        imgLR = sample['imageLR']
        mask = sample['label']
        if random.random()<self.p:
            rotate_degree = random.choice(self.degree)
            if rotate_degree != 0:
                img = img.rotate(rotate_degree)
                imgLR = imgLR.rotate(rotate_degree)
                mask = mask.rotate(rotate_degree)

        return {'image': img,
                'imageLR': imgLR,
                'label': mask}
