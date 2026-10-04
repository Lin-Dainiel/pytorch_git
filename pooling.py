import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms
import os
import cv2
from torch.nn import Conv2d


class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.label_dir = "portrait"
        self.img_name_list = os.listdir(self.label_dir)
        self.transform =  transform
    def __getitem__(self, item):
        img_name = self.img_name_list[item]
        img_name_split = os.path.splitext(img_name)[0]
        img_path_list = os.path.join(self.label_dir,img_name)
        img = Image.open(img_path_list)
        if self.transform:
            img = self.transform(img)
        return img, img_name_split
    def __len__(self):
        return len(self.img_name_list)

img_transforms_tensor = transforms.ToTensor()
img_initialize = MyDataset(transform = img_transforms_tensor)
img_dataloader = DataLoader(
    img_initialize,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

class MyConv(nn.Module):
    def __init__(self):
        super(MyConv, self).__init__()
        self.conv = Conv2d(3, 3, 3, 1,bias=False)
        kernel = torch.tensor([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=torch.float32)
        k = kernel.view(1, 1, 3, 3).repeat(3, 3, 1, 1)
        with torch.no_grad():
            self.conv.weight.copy_(k)
        self.conv.weight.requires_grad = False
    def forward(self, x):
        x = self.conv(x)
        return x

my_conv = MyConv()
writer = SummaryWriter("logs")
step = 0
for imgs, labels in img_dataloader:
    out_imgs = my_conv(imgs)
    writer.add_image("input",imgs.squeeze(0),step)# ('a', 'b', 'c')
    writer.add_image("output", out_imgs.squeeze(0), step)
    step += 1

writer.close()

# for i in range(5):
#     img_PIL = img_initialize[i][0]
#     img_tensor = img_transforms_tensor(img_PIL)
#     img_dataloader = DataLoader(img_tensor, batch_size=3, shuffle=True, num_workers=0)
#     print(img_tensor)
#     print(img_tensor.shape)
















