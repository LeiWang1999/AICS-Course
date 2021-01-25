# TF-Lenet

[![MIT License at http://tf-lenet.readthedocs.io/en/latest/license.html](https://img.shields.io/badge/license-MIT-blue.svg)](http://tf-lenet.readthedocs.io/en/latest/license.html)
[![Documentation at http://tf-lenet.readthedocs.io/en/latest/](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://tf-lenet.readthedocs.io/en/latest/)
[![Requirements Status](https://requires.io/github/ragavvenkatesan/tf-lenet/requirements.svg?branch=master)](https://requires.io/github/ragavvenkatesan/tf-lenet/requirements/?branch=master)
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)

* Are you a long time user of Theano and considering migrating to TensorFlow?
* Are you here to learn what CNNs are and how to implement them in TensorFlow?

Then you're in luck! I myself am just migrating from theano to tensorflow and this is my first implementation in 
TensorFlow. This is a typical Lenet-5 network trained to classify MNIST-like dataset.
This is a simple code that is found in the tutorial that comes with TensorFlow.
I made it a little modularized so that it could be re-purposed for other networks later.
The [documentation](http://tf-lenet.readthedocs.io) website that comes along with this repository helps users migrating from theano to tensorflow.

Obviously, you'd need [tensorflow](https://www.tensorflow.org/install/) and [numpy](https://docs.scipy.org/doc/numpy/user/install.html) installed. To run the code simply run:

```bash
python main.py
```

Once run, you can view all summaries at the tensorboard by running:

```bash
tensorboard --logdir tensorboard
```

Refer to the documentation site for more details.
Thanks for checking out the code, hope it was useful.

Ragav Venkatesan

http://www.ragav.net
