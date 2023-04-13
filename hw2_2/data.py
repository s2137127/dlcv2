from torch.utils.data import Dataset, DataLoader
import torch
import imageio.v2 as imageio
import os
import csv
from torchvision import transforms, datasets
import numpy as np
class Dataset(Dataset):
    def __init__(self, datatype, transform=None):
        self.filename_img = []
        self.transform = transform
        if datatype == 'train':
            self.path = '../hw2_data/hw2_data/digits/mnistm/train.csv'
        elif datatype == 'valid':
            self.path = '../hw2_data/hw2_data/digits/mnistm/val.csv'

        file = open(self.path)
        reader = csv.reader(file)
        header = next(reader)
        # print("head",header)
        self.filename_img = np.array(sorted([[img_name, label ]for (img_name,label) in reader],key=lambda s: s[1]))


        file.close()

    def __len__(self):
        return self.filename_img.shape[0]

    def __getitem__(self, idx):
        path = '../hw2_data/hw2_data/digits/mnistm/data'
        # print(self.filename_img[:,0])
        image = imageio.imread(os.path.join(path, self.filename_img[idx,0]))
        # print('label',self.filename_img[idx,:])
        if self.transform:
            image = self.transform(image)
        return image,int(self.filename_img[idx,1])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((16,16)),
    transforms.RandomResizedCrop((16,16), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

batch_size = 32
workers =2

dataset_t = Dataset(datatype='train', transform=transform)
dataset_v = Dataset(datatype='valid', transform=transform)
# Create the dataloader
dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=batch_size,
                                         shuffle=True, num_workers=workers,pin_memory=True)
dataloader_v = torch.utils.data.DataLoader(dataset_v, batch_size=batch_size,
                                         shuffle=True, num_workers=workers,pin_memory=True)

def get_dataloader():
    return dataloader_t,dataloader_v

