from __future__ import print_function
import sys, os, pdb
sys.path.insert(0, 'src')
import numpy as np, scipy.misc 
from optimize import optimize
from argparse import ArgumentParser
from utils import save_img, get_img, exists, list_files
import evaluate

def ordinary_least_squares(arr):
    if len(arr) == 0:
        return [0, 0, 0]
    maxValue = max(arr)

    x = np.arange(1, len(arr) + 1, 1)
    y = np.array(arr)
    z = np.polyfit(x, y, 1)

    p = np.poly1d(z)

    return [z[0], z[1], maxValue]

def get_growth(arr):
    growth, b, maxSale = ordinary_least_squares(arr)
    return growth

def get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

style_target = get_img('examples/style/rain_princess.jpg')
content_targets = get_files('data/train2014_test')
content_weight = 1.5e1
style_weight = 1e2
tv_weight = 2e2
vgg_path = 'data/imagenet-vgg-verydeep-19.mat'

def test_train():
    loss_arr = []

    for preds, losses, i, epoch in optimize(content_targets, style_target, content_weight, style_weight,
                 tv_weight, vgg_path, epochs=1, print_iterations=1,
                 batch_size=4, save_path='ckp_temp/fns.ckpt', slow=False,
                 learning_rate=1e-3, debug=False, type=0, save=False):
        style_loss, content_loss, tv_loss, loss = losses
        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        loss_arr.append(loss)
        to_print = (style_loss, content_loss, tv_loss)
        print('style: %s, content:%s, tv: %s' % to_print)

    if len(loss_arr) > 2:
        loss_arr.remove(min(loss_arr))
        loss_arr.remove(max(loss_arr))

    growth = get_growth(loss_arr)
    print('growth: %f'%growth)
    if growth >= 0:
        print('TEST TRAINING FAILED, LOSS IS NOT DECLINING.')
        exit(0)
    elif growth < 0:
        print('TEST TRAINING SUCCESS.')

def test_train_using_instance_norm():
    loss_arr = []

    for preds, losses, i, epoch in optimize(content_targets, style_target, content_weight, style_weight,
                 tv_weight, vgg_path, epochs=1, print_iterations=1,
                 batch_size=4, save_path='ckp_temp/fns.ckpt', slow=False,
                 learning_rate=1e-3, debug=False, type=1, save=True):
        style_loss, content_loss, tv_loss, loss = losses
        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        loss_arr.append(loss)
        to_print = (style_loss, content_loss, tv_loss)
        print('style: %s, content:%s, tv: %s' % to_print)

    if len(loss_arr) > 2:
        loss_arr.remove(min(loss_arr))
        loss_arr.remove(max(loss_arr))

    growth = get_growth(loss_arr)
    print('growth: %f'%growth)
    if growth >= 0:
        print('TEST TRAINING USING INSTANCE NORM FAILED, LOSS IS NOT DECLINING.')
        exit(0)
    elif growth < 0:
        print('TEST TRAINING USING INSTANCE NORM SUCCESS.')

def test_ckpt():
    evaluate.ffwd_to_img('data/chicago.jpg', 'out/test_result.jpg', 'ckp_temp/')
    print('EVALUATE FINSHED.')


def main():
    os.system('rm ckp_temp/*')
    test_train()
    print('----------------------')
    test_train_using_instance_norm()
    print('----------------------')
    test_ckpt()

if __name__ == '__main__':
    main()