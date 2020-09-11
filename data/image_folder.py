###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.txt'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    stop = 100000

    for root, _, fnames in sorted(os.walk(dir)):
        if stop == 0:
            break
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                stop -= 1
    # print images, len(images)
    return images

def make_dataset_unaligned(dir, stop=100000):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    print(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        # print fnames, dir, min(stop, len(fnames))
        random_perm = np.random.choice(len(fnames), min(stop, len(fnames)))
        for f_index in random_perm:
            fname = fnames[f_index]
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                stop -= 1
            if stop == 0:
                break
    # print len(images), stop
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
