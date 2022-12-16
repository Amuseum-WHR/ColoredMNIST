import os
import torch
from model import LeNet

from datasets import ColoredMNIST
from torch.utils.data import DataLoader
from tools import show_tensor, change_color

data_dir = '/my/datasets/path'
env = [0.1] # image with label = 0~4 is green with p of 0.9,and image with label = 5~9 is red with p of 0.9
colormnist = ColoredMNIST(data_dir, env) # contains len(env) dataset
dataset_env0 = colormnist[0]
train_loaders = DataLoader(dataset=dataset_env0, batch_size=1)

model = LeNet()
# model.train()
for i, (img, label) in enumerate(train_loaders):
    print(label)
    show_tensor(img[0])
    show_tensor(change_color(img)[0])
    break
