.. tf-lenet documentation master file, created by
   sphinx-quickstart on Wed Aug  9 09:29:33 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

A Gentle Intro to TensorFlow for Theano Users
=============================================
 
Welcome to the Lenet tutorial using TensorFlow. 
From being a long time user of `Theano <https://github.com/Theano/Theano>`_, migrating to `TensorFlow <https://www.tensorflow.org/>`_ is not that easy.
Recently, tensorflow is showing strong performance leading to many defecting from theano to tensorflow.
I am one such defector.
This repository contains an implementation Lenet, the hello world of deep CNNs  and is my first exploratory experimentation with TensorFlow.
It is a typical Lenet-5 network trained to classify MNIST dataset. 
This is a simple implementation similar to that, which can be found in the tutorial that comes with TensorFlow and most other public service tutorials.
This is however modularized, so that it is easy to understand and reuse.
This documentation website that comes along with this repository might help users migrating from theano to tensorflow, just as I did while 
implementing this repository. 
In this regard, whenever possible, I make explicit comparisons to help along. 
Tensorflow has many ``contrib`` packages that are a level of abstraction higher than theano. 
I have avoided using those whenever possible and stuck with the fundamental tensorflow modules for this tutorial.

.. image:: https://requires.io/github/ragavvenkatesan/tf-lenet/requirements.svg?branch=master
    :target: https://requires.io/github/ragavvenkatesan/tf-lenet/requirements/?branch=master
    :alt: Requirements Status

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: license.html
   :alt: MIT License

.. image:: https://img.shields.io/badge/contributions-welcome-green.svg   
    :target: https://github.com/ragavvenkatesan/tf-lenet/
    :alt: Fork to contribute to the GitHub codebase

.. image:: https://readthedocs.org/projects/tf-lenet/badge/?version=latest
    :target: http://tf-lenet.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status   

.. image:: https://badges.frapsoft.com/os/v1/open-source.svg?v=103
    :alt: Support Open Source

While this is most useful for theano to tensorflow migrants, this will also be useful for those who are new to CNNs.
There are small notes and materials explaining the theory and math behind the working of CNNs and layers.
While these are in no way comprehensive, these might help those that are unfamiliar with
CNNs but want to simply learn tensorflow and would rather not spend time on a semester long course.

.. note:: 

    The theoretical material in this tutorial are adapted from a forthcoming book chapter on *Feature Learning for Images*

To begin with, it might be helpful to run the entire code in its default setting.
This will enable you to ensure that the installations were proper and that your machine was setup.

Obviously, you'd need `tensorflow <https://www.tensorflow.org/install/>`_ and `numpy <https://docs.scipy.org/doc/numpy/user/install.html>`_ installed.
There might be other tools that you'd require for advanced uses which you can find in the ``requirements.txt`` file that ships along with this code.  
Firstly, clone the repository down into some directory as follows,

.. code-block:: bash

    git clone http://github.com/ragavvenkatesan/tf-lenet
    cd tf-lenet

You can then run the entire code in one of two ways.
Either run the ``main.py`` file like so:

.. code-block:: python 

    python main.py

or type in the contents of that file, line-by-line in a python shell:

.. code-block:: python 

    from lenet.trainer import trainer
    from lenet.network import lenet5      
    from lenet.dataset import mnist

    dataset = mnist()   
    net = lenet5(images = dataset.images)  
    net.cook(labels = dataset.labels)
    bp = trainer (net, dataset.feed)
    bp.train()

Once the code is running, setup tensorboard to observe results and outputs. 

.. code-block:: bash

    tensorboard --logdir=tensorboard

If everything went well, the tensorboard should have content populated in it.
Open a browser and enter the address ``0.0.0.0:6006``, this will open up tensorboard.
The accuracy graph in the scalars tab under the test column will look like the following:

.. figure:: tutorial/figures/accuracy.png
   :alt: Accuracy of the network once fully trained.

This implies that the network trained fully and has achieved about 99% accuracy and everything is normal.
From the next section onwards, I will go in detail, how I built this network. 

If you are interested please check out my `Yann Toolbox <http://yann.network>`_ written in theano completely.
Have fun! 

.. toctree::
   :maxdepth: 3
   :name: hidden  
   :hidden:   

   tutorial/index
   api/index
   license