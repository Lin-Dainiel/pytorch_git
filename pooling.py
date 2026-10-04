import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os


# image_path = "portrait/1-seated.jpg"
# image = Image.open(image_path)
# print(image)
# image.show()

class MyDataset(Dataset):
    def __init__(self):
        self.label_dir = "portrait"
        self.img_name_list = os.listdir(self.label_dir)
    def __getitem__(self, item):
        img_name = self.img_name_list[item]
        img_name_split = os.path.splitext(img_name)[0]
        img_path_list = os.path.join(self.label_dir,img_name)
        img = Image.open(img_path_list)
        return img, img_name_split

# image_initialize = MyDataset()
# print(image_initialize[3])
# image_initialize[3][0].show()




