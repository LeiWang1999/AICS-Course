Reshaping layers
================

Most often datasets are of the form where :math:`\mathbf{x} \in [x_0,x_1, \dots x_d]` of :math:`d` dimensions.
We want :math:`\mathbf{x} \in \mathbb{R}^{\sqrt{d} \times \sqrt{d}}` for the convolutional layers.
For this and for other purposes, we might be helped by having some reshaping layers. 
Here are a few in tensorflow.

For flattening, which is converting an image into a vector, used for instance, before feeding into the first 
fully-connected layers:

.. code-block:: python

    in_shp = input.get_shape().as_list()
    output = tf.reshape(input, [-1, in_shp[1]*in_shp[2]*in_shp[3]])

The reshape command is quite simlar to theano. The nice thing here is the ``-1`` option, which implies that 
any dimension that have ``-1`` immediately accommodates the rest. This means that we don't have to care about 
knowing the value of that dimension and could assign it during runtime. 
I use this for the mini batch size being unknown. One network can now run in batch, stochastic or online 
gradient descent and during test time, I can supply how many ever samples I want. 

Similarly, we can also implement an unflatten layer:

.. code-block:: python

    dim = int( np.sqrt( input.shape[1].value / channels ) ) 
    output = tf.reshape(input, [-1, dim, dim, channels])    

These are also found in the :ref:`layers` module in :meth:`lenet.layers.flatten_layer` and :meth:`lenet.layers.unflatten_layer`.
