import os
import os.path

import torch.utils.data as data
from PIL import Image


def make_dataset(is_supervised):
    input = open('./train_input.txt')
    ground_t = open('./train_gt.txt')
    depth_t = open('./train_depth.txt')
    if is_supervised:
        real_t = open('./train_input.txt')
    else:
        real_t = open('./train_real.txt')
    image = [(os.path.join(img_name.strip('\n'))) for img_name in
             input]
    gt = [(os.path.join(img_name.strip('\n'))) for img_name in
             ground_t]
    depth = [(os.path.join(img_name.strip('\n'))) for img_name in
          depth_t]
    real = [(os.path.join(img_name.strip('\n'))) for img_name in
             real_t]

    input.close()
    ground_t.close()
    depth_t.close()
    real_t.close()
    return [[image[i], gt[i], depth[i],real[i]]for i in range(len(image))]



class ImageFolder(data.Dataset):
    def __init__(self, triple_transform=None, transform=None, target_transform=None, is_supervised=True):
        self.imgs = make_dataset(is_supervised)
        self.triple_transform = triple_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, depth_path,real_path = self.imgs[index]
        img = Image.open(img_path)
        target = Image.open(gt_path)
        depth = Image.open(depth_path)
        real = Image.open(real_path)
        if self.triple_transform is not None:
            img, target, depth,real = self.triple_transform(img, target, depth,real)
        if self.transform is not None:
            img = self.transform(img)
            real = self.transform(real)
        if self.target_transform is not None:
            target = self.target_transform(target)
            depth = self.target_transform(depth)


        return img, target, depth,real

    def __len__(self):
        return len(self.imgs)
