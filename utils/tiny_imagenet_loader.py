import numpy as np
import torch
from bisect import bisect_left
from PIL import Image
import random
import os

class TinyImagenet(torch.utils.data.Dataset):

    def __init__(self, transform=None):

        data_path_file = "/home/leiyvtian/ood-cls/datasets/scood/data/imglist/benchmark_cifar10/train_tin.txt"
        with open(data_path_file,'r') as f:
            self.data_path = f.readlines()
            self.data_path = [item.strip('\n').split(' ')[0] for item in self.data_path]
        f.close()
        ### 统一random，在训练中固定seed的位置一定是后于dataloader的
        random.seed(0)
        self.data_path = random.sample(self.data_path, len(self.data_path))
        self.offset = 0     # offset index

        self.transform = transform
        self.temp = []

    def __getitem__(self, index):
        index = (index + self.offset) % 100000
        # img = self.load_image(index)
        img = Image.open(os.path.join("/home/leiyvtian/ood-cls/datasets/scood/data/images/", self.data_path[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # print(self.data_path)
        if img.shape[0]!=3:
            print(self.data_path[index])
        return img, 0  # 0 is the class

    def __len__(self):
        return 100000
    

class TinsetImageNet32(torch.utils.data.Dataset):

    def __init__(self, transform=None):

        self.data_basedir = "/home/leiyvtian/ood-cls/datasets/unlabeled_datasets/TinsetImageNet32"
        # with open(data_path_file,'r') as f:
        #     self.data_path = f.readlines()
        #     self.data_path = [item.strip('\n').split(' ')[0] for item in self.data_path]
        # f.close()
        self.data_path = []
        for dir in os.listdir(self.data_basedir):
            dirs = [os.path.join(self.data_basedir, dir, path) for path in os.listdir(os.path.join(self.data_basedir, dir))]
            # dirs = random.sample(dirs)
            self.data_path += dirs
        ### 统一random，在训练中固定seed的位置一定是后于dataloader的
        random.seed(0)
        self.data_path = random.sample(self.data_path, len(self.data_path))
        self.offset = 0     # offset index

        self.transform = transform
        self.temp = []

    def __getitem__(self, index):
        index = (index + self.offset) % len(self.data_path)
        # img = self.load_image(index)
        img = Image.open(self.data_path[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # print(self.data_path)
        # if img.shape[0]!=3:
        #     print(self.data_path[index])
        return img, 0  # 0 is the class

    def __len__(self):
        return len(self.data_path)
