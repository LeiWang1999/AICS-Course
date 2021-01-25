Normalization Layer
===================

Implementing normalization was much simpler than using theano. All I had to do was 

.. code-block:: python

    output = tf.nn.lrn(input)

:meth:`tf.nn.lrn` is an implementation of the local response normalization :cite:`lyu2008nonlinear`. 
This layer definition could also be found in the :meth:`lenet.layers.local_response_normalization_layer` method.