from model import DANN
import torch
from sklearn import manifold
from data import *
import matplotlib.pyplot as plt
feature_dic = {}
def get_activation(name):
    def hook(model, input, output):
        feature_dic[name] = output.detach()

    return hook
def tsne(data_name):
    _,source_v = get_dataloader_mnistm(batch_size=1)
    if data_name == 'svhn':
        dann = DANN()
        dann.load_state_dict(torch.load('./dann_ms.pth'))
        _,target_v = get_dataloader_svhn(batch_size=1)
    else:
        dann = DANN()
        dann.load_state_dict(torch.load('./dann_mu.pth'))
        _, target_v = get_dataloader_usps(batch_size=1)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dann.cuda()
    dann.eval()
    print('Length of val Set:', len(target_v))
    label1_arr = []
    label2_arr = []
    out_arr = []
    out_arr2 = []
    print(dann)
    with torch.no_grad():
        handle = dann.class_classifier.c_fc2.register_forward_hook(get_activation('c_features'))
        handle2 = dann.domain_classifier.d_fc1.register_forward_hook(get_activation('d_features'))
        alpha=0
        for (x, label) in target_v:
            if use_cuda:
                x = x.cuda()

            y = dann(x,alpha)
            # print(label.size(0))
            label1_arr.append(label[0])
            label2_arr.append(0)
            # for i in range(label.size(0)):

            out_arr.append(feature_dic['c_features'].to("cpu").numpy()[0])
            out_arr2.append(feature_dic['d_features'].to("cpu").numpy()[0])

        for (x, label) in source_v:
            if use_cuda:
                x = x.cuda()

            y = dann(x,alpha)
            # print(label.size(0))
            label1_arr.append(label[0])
            label2_arr.append(1)
            # for i in range(label.size(0)):

            out_arr.append(feature_dic['c_features'].to("cpu").numpy()[0])
            out_arr2.append(feature_dic['d_features'].to("cpu").numpy()[0])

    handle.remove()
    handle2.remove()

    feature = np.array(out_arr.copy())
    feature2 = np.array(out_arr.copy())
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=0).fit_transform(feature)
    X_tsne2 = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=0).fit_transform(feature2)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # Normalize
    x_min2, x_max2 = X_tsne2.min(0), X_tsne2.max(0)
    X_norm2 = (X_tsne2 - x_min2) / (x_max2 - x_min2)  # Normalize
    # print(X_norm.shape)
    # print(label_arr[2].item())
    fig = plt.figure(figsize=(16, 8))
    # print("sss",X_norm.shape)
    ax1 = fig.add_subplot(121)
    ax1.set_title("by class")
    ax1.scatter(X_norm[:, 0], X_norm[:, 1], c=label1_arr, cmap='rainbow')
    # for i in range(X_norm.shape[0]):
    #     ax1.annotate(str(label_arr[i].item()), xy=(X_norm[i, 0], X_norm[i, 1]), xytext=(X_norm[i, 0], X_norm[i, 1]),
    #                  alpha=0.1, annotation_clip=True)
    ax2 = fig.add_subplot(122)
    ax2.set_title("by domain")
    ax2.scatter(X_norm2[:, 0], X_norm2[:, 1], c=label2_arr, cmap='rainbow')
    # for i in range(X_norm2.shape[0]):
    #     ax2.annotate(str(label_arr[i].item()), xy=(X_norm2[i, 0], X_norm2[i, 1]), xytext=(X_norm2[i, 0], X_norm2[i, 1]),
    #                  alpha=0.1, annotation_clip=True)
    plt.savefig('data_visdualize_svhn.jpg')

    plt.show()
if __name__ == '__main__':
    tsne('svhn')
