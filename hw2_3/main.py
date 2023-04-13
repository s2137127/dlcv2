#https://github.com/fungtion/DANN

import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from model import *
from data import *
import numpy as np

manual_seed = 666
random.seed(manual_seed)
torch.manual_seed(manual_seed)
config = {
    'lr' : 1e-3,
    'batch_size' : 128,
    'image_size': 28,
    'epochs':80,
    'device':'cuda' if torch.cuda.is_available() else 'cpu',
    'best_acc':0
}

def train_dann(data_loader,domain):
    dann = DANN().to(config['device'])
    source_t,source_v,target_t,target_v = data_loader
    optimizer = optim.Adam(dann.parameters(), lr=config['lr'])
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    loss_class = loss_class.to(config['device'])
    loss_domain = loss_domain.to(config['device'])
    for p in dann.parameters():
        p.requires_grad = True

    for epoch in range(config['epochs']):

        len_dataloader = min(len(source_t), len(target_t))
        data_source_iter = iter(source_t)
        data_target_iter = iter(target_t)

        i = 0
        while i < len_dataloader:

            p = float(i + epoch * len_dataloader) / config['epochs'] / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = data_source_iter.next()
            s_img, s_label = data_source
            s_img = s_img.to(config['device'])
            s_label = s_label.to(config['device'])
            dann.zero_grad()
            batch_size = len(s_label)

            input_img = torch.FloatTensor(batch_size, 3, config['image_size'], config['image_size']).to(config['device'])
            class_label = torch.LongTensor(batch_size).to(config['device'])
            domain_label = torch.zeros(batch_size).long().to(config['device'])

            input_img.resize_as_(s_img).copy_(s_img)
            class_label.resize_as_(s_label).copy_(s_label)

            class_output, domain_output = dann(input_data=input_img, alpha=alpha)
            err_s_label = loss_class(class_output, class_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            data_target = data_target_iter.next()
            t_img, _ = data_target
            t_img = t_img.to(config['device'])
            batch_size = len(t_img)

            input_img = torch.FloatTensor(batch_size, 3, config['image_size'], config['image_size']).to(config['device'])
            domain_label = torch.ones(batch_size)
            domain_label = domain_label.long().to(config['device'])

            input_img.resize_as_(t_img).copy_(t_img)

            _, domain_output = dann(input_data=input_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            i += 1
            if i%100 == 0 or i == len_dataloader-1:
                print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                   err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

####################################################333
###################valid################
        dann.eval()
        with torch.no_grad():
            alpha=0
            n_correct = 0
            for t_img, t_label in target_v:

                t_img = t_img.cuda()
                t_label = t_label.cuda()

                input_img = torch.FloatTensor(len(t_label), 3, config['image_size'], config['image_size']).to(config['device'])
                class_label = torch.LongTensor(len(t_label)).to(config['device'])

                input_img.resize_as_(t_img).copy_(t_img)
                class_label.resize_as_(t_label).copy_(t_label)

                class_output, _ = dann(input_data=input_img, alpha=alpha)
                pred = class_output.data.max(1, keepdim=True)[1]
                n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()

            accu = n_correct.data.numpy() * 1.0 / len(target_v)


            if accu > config['best_acc']:
                config['best_acc'] = accu
                torch.save(dann.state_dict(),'./dann_%s.pth' %domain)

            print('epoch: %d, acc : %f ,best_acc = %f' % (epoch, accu,config['best_acc']))
        dann.train()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("train mnistm to svhn")
    train_dann(get_dataloader_ms(config['batch_size']),'ms')
    print("train mnistm to usps")
    train_dann(get_dataloader_mu(config['batch_size']), 'mu')

