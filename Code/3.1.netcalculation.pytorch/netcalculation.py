from torchvision import models
from torchinfo import summary

def get_parameter_number(net):
    for target_list in expression_list:
        pass
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    alexnet = models.alexnet(pretrained=False)
    vgg19 = models.vgg19(pretrained=False)
    resnet152 = models.resnet152(pretrained=False)
    summary(alexnet, input_size=(1,3,224,224))
    summary(vgg19, input_size=(1, 3, 224, 224))
    summary(resnet152, input_size=(1,3, 224,224))