import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import torchvision.utils as vutils

dataroot = '../hw2_data/hw2_data/face/'
workers = 2

# Batch size during training
batch_size = 128
image_size = 64






class Dataset(Dataset):
    def __init__(self, datatype, transform=None):
        self.filename_img = []
        self.transform = transform
        if datatype == 'train':
            self.path = '../hw2_data/hw2_data/face/train'
            self.filename_img = sorted([file for file in os.listdir(self.path)
                                        if file.endswith('.png')])
        else:
            self.path = '../hw2_data/hw2_data/face/val'
            self.filename_img = sorted([file for file in os.listdir(self.path)
                                        if file.endswith('.png')])

    def __len__(self):
        return len(self.filename_img)

    def __getitem__(self, idx):
        image = imageio.imread(os.path.join(self.path, self.filename_img[idx]))
        if self.transform:

            image = self.transform(image)
            image = torch.tanh(image)
        return image



# dataset = datasets.ImageFolder(root=dataroot,
#                            transform=transforms.Compose([
#                                transforms.Resize(image_size),
#                                transforms.CenterCrop(image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = Dataset(datatype='train', transform=transform)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
# plt.show()

def get_dataloader():
    return dataloader
