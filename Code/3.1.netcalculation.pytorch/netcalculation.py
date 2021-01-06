import torch
from torchvision import models


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    alexnet = models.alexnet(pretrained=False)
    res = get_parameter_number(alexnet)
    print(res)