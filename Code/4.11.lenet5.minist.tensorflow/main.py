from lenet.trainer import trainer
from lenet.network import lenet5      
from lenet.dataset import mnist, fashion_mnist

if __name__ == '__main__':
    # dataset = mnist()         # Use this for classic MNIST
    dataset = fashion_mnist()   # Use this for the new fashion MNIST
    net = lenet5(images = dataset.images)  
    net.cook(labels = dataset.labels)
    bp = trainer (net, dataset.feed)
    bp.train()