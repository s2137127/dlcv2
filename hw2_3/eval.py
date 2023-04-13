from model import *
from torch.utils.data import Dataset, DataLoader

from data import *
config = {

    'batch_size' : 128,
    'image_size': 28,

    'device':'cuda' if torch.cuda.is_available() else 'cpu',

}
workers = 2
def eval(data):

    n_correct = 0
    alpha=0
    if data == 'svhn':
        dann = DANN().to(config['device'])
        dann.load_state_dict(torch.load('../dann_ms.pth'))
        target_v = torch.utils.data.DataLoader(svhn_v, batch_size=config['batch_size'],
                                    shuffle=True, num_workers=workers)
        dann.eval()
        out=[]
        with torch.no_grad():
            cnt = 0
            for t_img,t_label in target_v:
                cnt += len(t_label)
                t_img = t_img.cuda()
                t_label = t_label.cuda()

                input_img = torch.FloatTensor(len(t_label), 3, config['image_size'], config['image_size']).to(
                    config['device'])
                class_label = torch.LongTensor(len(t_label)).to(config['device'])

                input_img.resize_as_(t_img).copy_(t_img)
                class_label.resize_as_(t_label).copy_(t_label)

                class_output, _ = dann(input_data=input_img, alpha=alpha)
                pred = class_output.data.max(1, keepdim=True)[1]


                n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()

            accu = n_correct.data.numpy() * 1.0 / cnt
            print('accuracy for svhn:', accu,"correct",n_correct.data.numpy(),"total",cnt)


    else:
        dann = DANN().to(config['device'])
        dann.load_state_dict(torch.load('../dann_mu.pth'))
        target_v = torch.utils.data.DataLoader(usps_v, batch_size=config['batch_size'],
                                               shuffle=True, num_workers=workers)
        n_correct = 0
        dann.eval()
        with torch.no_grad():
            cnt = 0
            for t_img, t_label in target_v:
                cnt += len(t_label)
                t_img = t_img.cuda()
                t_label = t_label.cuda()

                input_img = torch.FloatTensor(len(t_label), 3, config['image_size'], config['image_size']).to(
                    config['device'])
                class_label = torch.LongTensor(len(t_label)).to(config['device'])

                input_img.resize_as_(t_img).copy_(t_img)
                class_label.resize_as_(t_label).copy_(t_label)

                class_output, _ = dann(input_data=input_img, alpha=alpha)
                pred = class_output.data.max(1, keepdim=True)[1]
                n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()

            accu = n_correct.data.numpy() * 1.0 / cnt
            print('accuracy for usps:', accu)

def eval2(data):

    n_correct = 0
    size = 0
    if data == 'svhn':
        cnn = CNN().to(config['device'])
        cnn.load_state_dict(torch.load('../dann_mnist.pth'))
        target_v = torch.utils.data.DataLoader(svhn_v, batch_size=config['batch_size'],
                                    shuffle=True, num_workers=workers)

        for t_img,t_label in target_v:

            t_img = t_img.cuda()
            t_label = t_label.cuda()

            input_img = torch.FloatTensor(len(t_label), 3, config['image_size'], config['image_size']).to(
                config['device'])
            class_label = torch.LongTensor(len(t_label)).to(config['device'])

            input_img.resize_as_(t_img).copy_(t_img)
            class_label.resize_as_(t_label).copy_(t_label)

            class_output= cnn(input_data=input_img)
            pred = class_output.data.max(1, keepdim=True)[1]
            size += len(pred)
            n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()

        accu = n_correct.data.numpy() * 1.0 / size
        print('accuracy for svhn:', accu)


    else:
        cnn = CNN().to(config['device'])
        cnn.load_state_dict(torch.load('../dann_mnist.pth'))
        target_v = torch.utils.data.DataLoader(usps_v, batch_size=config['batch_size'],
                                               shuffle=True, num_workers=workers)
        n_correct = 0
        size = 0
        for t_img, t_label in target_v:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

            input_img = torch.FloatTensor(len(t_label), 3, config['image_size'], config['image_size']).to(
                config['device'])
            class_label = torch.LongTensor(len(t_label)).to(config['device'])

            input_img.resize_as_(t_img).copy_(t_img)
            class_label.resize_as_(t_label).copy_(t_label)

            class_output= cnn(input_data=input_img)
            pred = class_output.data.max(1, keepdim=True)[1]
            size += len(pred)
            n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()

        accu = n_correct.data.numpy() * 1.0 / size
        print('accuracy for usps:', accu)
if __name__ == '__main__':
    eval('svhn')
    print('1')
    eval('usps')
    print('2')

