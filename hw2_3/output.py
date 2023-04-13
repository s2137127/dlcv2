import csv
import os

import imageio.v2 as imageio
import torch
from torchvision.transforms import transforms
import numpy as np
from model import *
from torch.utils.data import Dataset, DataLoader
from os import mkdir
from os.path import isdir, dirname, basename
import sys
# from data import *
config = {

    'batch_size' : 128,
    'image_size': 28,

    'device':'cuda' if torch.cuda.is_available() else 'cpu',

}
workers = 2
class Dataset(Dataset):
    def __init__(self ,path, transform=None):
        self.filename_img = []
        self.transform = transform
        self.data = None

        self.path = path
        self.filename_img = sorted([name for name in os.listdir(self.path)
                                    if name.endswith('.png')])

    def __len__(self):
        return len(self.filename_img)

    def __getitem__(self, idx):

        image = imageio.imread(os.path.join(self.path, self.filename_img[idx]))
        # print('label',self.filename_img[idx,:])
        if 'usps' in self.path:
            self.data = 'usps'
            image = self.gray2rgb(image)
        else:
            self.data = 'svhn'

        if self.transform:
            image = self.transform(image)

        return image,self.filename_img[idx]

    def gray2rgb(self,image):
        out = torch.tensor(np.array((image,image,image)))
        return out
def eval(dataset):
    if dataset.data == 'svhn':
        dann = DANN().to(config['device'])
        dann.load_state_dict(torch.load('./dann_ms.pth'))
        target_v = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'],
                                    shuffle=True,num_workers=workers)

    else:
        dann = DANN().to(config['device'])
        dann.load_state_dict(torch.load('./dann_mu.pth'))
        target_v = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'],
                                               shuffle=True, num_workers=workers)
    dann.eval()
    with torch.no_grad():
        out = []
        for t_img, img_name in target_v:
            t_img = t_img.cuda()
            input_img = torch.FloatTensor(len(img_name), 3, config['image_size'], config['image_size']).to(
                config['device'])

            input_img.resize_as_(t_img).copy_(t_img)

            class_output, _ = dann(input_data=input_img, alpha=0)
            pred = class_output.data.max(1)[1]
            # print(pred.shape)
            for i in range(len(pred)):
                out.append((img_name[i],pred[i].item()))
        return out
img_transform_target = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
if __name__ == '__main__':

    data_path, output = sys.argv[1], sys.argv[2]
    dataset = Dataset(data_path,transform=img_transform_target)

    out = eval(dataset)

    if not isdir(dirname(output)):
        mkdir(dirname(output))
    with open(output, 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'label'])
        for i in out:
            writer.writerow([i[0], i[1]])

