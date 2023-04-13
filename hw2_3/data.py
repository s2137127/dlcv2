from torch.utils.data import Dataset, DataLoader
import torch
import imageio.v2 as imageio
import os
import csv
from torchvision import transforms, datasets
import numpy as np
class Dataset(Dataset):
    def __init__(self, datatype,data, transform=None):
        self.filename_img = []
        self.transform = transform
        self.data = data
        if datatype == 'train':
            self.path = '../hw2_data/hw2_data/digits/%s/train.csv' %self.data
        elif datatype == 'valid':
            self.path = '../hw2_data/hw2_data/digits/%s/val.csv' %self.data

        file = open(self.path)
        reader = csv.reader(file)
        header = next(reader)
        # print("head",header)
        self.filename_img = np.array(sorted([[img_name, label ]for (img_name,label) in reader],key=lambda s: s[1]))


        file.close()

    def __len__(self):
        return self.filename_img.shape[0]

    def __getitem__(self, idx):
        path = '../hw2_data/hw2_data/digits/%s/data' %self.data
        # print(self.filename_img[:,0])
        image = imageio.imread(os.path.join(path, self.filename_img[idx,0]))
        # print('label',self.filename_img[idx,:])
        if self.data == 'usps':
            image = self.gray2rgb(image)
        if self.transform:
            image = self.transform(image)

        return image,int(self.filename_img[idx,1])

    def gray2rgb(self,image):
        out = torch.tensor(np.array((image,image,image)))
        # out[:,:,:] =image
        return out
workers =2

img_transform_source = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

img_transform_target = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
mnist_t = Dataset(datatype='train',data='mnistm', transform=img_transform_source)
svhn_t = Dataset(datatype='train',data='svhn',  transform=img_transform_target)
usps_t = Dataset(datatype='train',data='usps', transform=img_transform_target)
mnist_v = Dataset(datatype='valid',data='mnistm', transform=img_transform_source)
svhn_v = Dataset(datatype='valid',data='svhn',  transform=img_transform_target)
usps_v = Dataset(datatype='valid',data='usps', transform=img_transform_target)
# Create the dataloader

def get_dataloader_mnistm(batch_size):
    dataloader_mt = torch.utils.data.DataLoader(mnist_t, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)
    dataloader_mv = torch.utils.data.DataLoader(mnist_v, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)

    return dataloader_mt,dataloader_mv


def get_dataloader_svhn(batch_size):
    dataloader_st = torch.utils.data.DataLoader(svhn_t, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)
    dataloader_sv = torch.utils.data.DataLoader(svhn_v, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)

    return dataloader_st, dataloader_sv

def get_dataloader_usps(batch_size):
    dataloader_ut = torch.utils.data.DataLoader(usps_t, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)
    dataloader_uv = torch.utils.data.DataLoader(usps_v, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)

    return dataloader_ut, dataloader_uv


def get_dataloader_ms(batch_size):
    dataloader_mt = torch.utils.data.DataLoader(mnist_t, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)
    dataloader_mv = torch.utils.data.DataLoader(mnist_v, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)
    dataloader_st = torch.utils.data.DataLoader(svhn_t, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)
    dataloader_sv = torch.utils.data.DataLoader(svhn_v, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)

    return dataloader_mt,dataloader_mv,dataloader_st,dataloader_sv

def get_dataloader_mu(batch_size):
    dataloader_ut = torch.utils.data.DataLoader(usps_t, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)
    dataloader_uv = torch.utils.data.DataLoader(usps_v, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)
    dataloader_mt = torch.utils.data.DataLoader(mnist_t, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)
    dataloader_mv = torch.utils.data.DataLoader(mnist_v, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    return dataloader_mt, dataloader_mv, dataloader_ut, dataloader_uv