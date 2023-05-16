import base64
import json
from io import BytesIO

import PIL.Image as Image
import torch
from torch import nn


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if 'is_training' in net.__code__.co_varnames:  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, loss, epochs, device, optimizer):
    net = net.to(device)
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc))


def padding_fit(pil_img, bg_color, target_size):
    width, height = pil_img.size
    if width == height:
        return pil_img.resize((target_size, target_size))
    else:
        size = max(width, height)
        res = Image.new(pil_img.mode, (size, size), bg_color)
        res.paste(pil_img, (0, abs(width - height) // 2))
        return res.resize((target_size, target_size))


def center_cut_fit(pil_img, target_size):
    width, height = pil_img.size
    if width == height:
        return pil_img.resize((target_size, target_size), Image.ANTIALIAS)
    else:
        res = pil_img
        if width >= height:
            res = res.resize((int(width * (target_size / width)), target_size), Image.ANTIALIAS)
            box = ((width - target_size) // 2, 0, (width - target_size) // 2 + target_size, target_size)
            return res.crop(box)
        else:
            res = res.resize((target_size, int(height * (target_size / height))), Image.ANTIALIAS)
            box = (0, (height - target_size) // 2, target_size, (height - target_size) // 2 + target_size)
            return res.crop(box)


def resize_fit(pil_img, target_size):
    return pil_img.resize((target_size, target_size), Image.LANCZOS)


def vgg_blk(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, 2)
    )


def cut_img(pil_img, left_x, width, height):
    box = (left_x, 0, left_x + width, height)
    return pil_img.crop(box)


def load_labels(file_path):
    with open(file_path) as label_file:
        label_dict = json.load(label_file)
    label_data = [i for i in range(len(label_dict))]
    for k, v in label_dict.items():
        label_data[v] = k
    return label_data


def base64_pil(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    return image
