
import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from model import *
from data import *
from tqdm import tqdm

manual_seed = 666
random.seed(manual_seed)
torch.manual_seed(manual_seed)
config = {
    'lr' : 1e-3,
    'batch_size' : 128,
    'image_size': 28,
    'epochs':15,
    'device':'cuda' if torch.cuda.is_available() else 'cpu',
    'best_acc':0
}

def train_cnn(data_loader,domain):
    cnn = CNN().to(config['device'])
    config['best_acc'] = 0
    optimizer = optim.Adam(cnn.parameters(), lr=config['lr'])
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    loss_class = loss_class.to(config['device'])
    train_ld,valid_ld = data_loader

    for p in cnn.parameters():
        p.requires_grad = True

    for epoch in range(config['epochs']):
        for i,(image,label) in enumerate(tqdm(train_ld)):

            image = image.to(config['device'])
            label = label.to(config['device'])
            cnn.zero_grad()
            batch_size = len(label)

            input_img = torch.FloatTensor(batch_size, 3, config['image_size'], config['image_size']).to(config['device'])
            class_label = torch.LongTensor(batch_size).to(config['device'])

            input_img.resize_as_(image).copy_(image)
            class_label.resize_as_(label).copy_(label)

            class_output = cnn(input_data=input_img)
            err_s_label = loss_class(class_output, class_label)

            err_s_label.backward()
            optimizer.step()

            if i%100 == 0 or i == len(data_loader)-1:
                print('epoch: %d, err_s_label: %f '% (epoch,  err_s_label.cpu().data.numpy()))


####################################################333
###################valid################
        cnn.eval()
        with torch.no_grad():
            n_correct = 0
            batch_size = 0
            for t_img, t_label in valid_ld:

                t_img = t_img.cuda()
                t_label = t_label.cuda()

                input_img = torch.FloatTensor(len(t_label), 3, config['image_size'], config['image_size']).to(config['device'])
                class_label = torch.LongTensor(len(t_label)).to(config['device'])

                input_img.resize_as_(t_img).copy_(t_img)
                class_label.resize_as_(t_label).copy_(t_label)

                class_output = cnn(input_data=input_img)
                pred = class_output.data.max(1, keepdim=True)[1]
                n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
                batch_size += len(t_label)
            accu = n_correct.data.numpy() * 1.0 / batch_size


            if accu > config['best_acc']:
                config['best_acc'] = accu
                torch.save(cnn.state_dict(),'./dann_%s.pth' %domain)

            print('epoch: %d, acc : %f ,best_acc = %f' % (epoch, accu,config['best_acc']))
        cnn.train()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("train mnistm")
    train_cnn(get_dataloader_mnistm(config['batch_size']),'mnist')
    print("train svhn ")
    train_cnn(get_dataloader_svhn(config['batch_size']), 'svhn')
    print("train usps")
    train_cnn(get_dataloader_usps(config['batch_size']), 'usps')