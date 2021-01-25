import tensorflow as tf
from .support import initializer, visualize_filters
import numpy as np

def softmax_layer (input, name = 'softmax'):
    """
    Creates the softmax normalization

    Args: 
        input: Where is the input of the layer coming from
        name: Name scope of the layer

    Returns:
        tuple: ``(softmax, prediction)``, A softmax output node and prediction output node
    """
    with tf.variable_scope(name) as scope:        
        inference = tf.nn.softmax(input, name = 'inference')
        predictions = tf.argmax(inference, 1, name = 'predictions')
    return (inference, predictions)

def dot_product_layer(input, params = None, neurons = 1200, name = 'fc', activation = 'relu'):
    """
    Creates a fully connected layer

    Args:
        input: Where is the input of the layer coming from
        neurons: Number of neurons in the layer.
        params: List of tensors, if supplied will use those params.
        name: name scope of the layer
        activation: What kind of activation to use.

    Returns:
        tuple: The output node and A list of parameters that are learnable
    """
    with tf.variable_scope(name) as scope:
        if params is None:
            weights = tf.Variable(initializer([input.shape[1].value,neurons], name = 'xavier_weights'),\
                                            name = 'weights')
            bias = tf.Variable(initializer([neurons], name = 'xavier_bias'), name = 'bias')
        else:
            weights = params[0]
            bias = params[1]

        dot = tf.nn.bias_add(tf.matmul(input, weights, name = 'dot'), bias, name = 'pre-activation')
        if activation == 'relu':
            activity = tf.nn.relu(dot, name = 'activity' )
        elif activation == 'sigmoid':
            activity = tf.nn.sigmoid(dot, name = 'activity' )            
        elif activation == 'identity':
            activity = dot                     
        params = [weights, bias]
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('bias', bias)  
        tf.summary.histogram('activity', activity)                              
    return (activity, params)

def conv_2d_layer (input, 
                neurons = 20,
                filter_size = (5,5), 
                stride = (1,1,1,1), 
                padding = 'VALID',
                name = 'conv', 
                activation = 'relu',
                visualize = False):
    """
    Creates a convolution layer

    Args:
        input: (NHWC) Where is the input of the layer coming from
        neurons: Number of neurons in the layer.
        name: name scope of the layer
        filter_size: A tuple of filter size ``(5,5)`` is default.
        stride: A tuple of x and y axis strides. ``(1,1,1,1)`` is default.
        name: A name for the scope of tensorflow
        visualize: If True, will add to summary. Only for first layer at the moment.
        activation: Activation for the outputs.
        padding: Padding to be used in convolution. "VALID" is default.

    Returns:
        tuple: The output node and A list of parameters that are learnable
    """        
    f_shp = [filter_size[0], filter_size[1], input.shape[3].value, neurons]
    with tf.variable_scope(name) as scope:
        weights = tf.Variable(initializer(  f_shp, 
                                            name = 'xavier_weights'),\
                                            name = 'weights')
        bias = tf.Variable(initializer([neurons], name = 'xavier_bias'), name = 'bias')
        c_out = tf.nn.conv2d(   input = input,
                                filter = weights,
                                strides = stride,
                                padding = padding,
                                name = scope.name  )
        c_out_bias = tf.nn.bias_add(c_out, bias, name = 'pre-activation')
        if activation == 'relu':
            activity = tf.nn.relu(c_out_bias, name = 'activity' )
        elif activation == 'sigmoid':
            activity = tf.nn.sigmoid(c_out_bias, name = 'activity' )            
        elif activation == 'identity':
            activity = c_out_bias
        params = [weights, bias]
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('bias', bias)  
        tf.summary.histogram('activity', activity) 
        if visualize is True:  
            visualize_filters(weights, name = 'filters_' + name)
    return (activity, params)        

def flatten_layer (input, name = 'flatten'):
    """
    This layer returns the flattened output
    Args:
        input: a 4D node.
        name: name scope of the layer.
    Returns:
        tensorflow tensor: a 2D node.
    """
    with tf.variable_scope(name) as scope:
        in_shp = input.get_shape().as_list()
        output = tf.reshape(input, [-1, in_shp[1]*in_shp[2]*in_shp[3]])
    return output 

def max_pool_2d_layer  (   input, 
                        pool_size = (1,2,2,1),
                        stride = (1,2,2,1),
                        padding = 'VALID',
                        name = 'pool' ):
    """
    Creates a max pooling layer

    Args:
        input: (NHWC) Where is the input of the layer coming from
        name: name scope of the layer
        pool_size: A tuple of filter size ``(5,5)`` is default.
        stride: A tuple of x and y axis strides. ``(1,1,1,1)`` is default.
        name: A name for the scope of tensorflow
        padding: Padding to be used in convolution. "VALID" is default.

    Returns:
        tensorflow tensor: The output node 
    """       
    with tf.variable_scope(name) as scope:
        output = tf.nn.max_pool (   value = input,
                                    ksize = pool_size,
                                    strides = stride,
                                    padding = padding,
                                    name = name ) 
    return output

def local_response_normalization_layer (input, name = 'lrn'):
    """
    This layer returns the flattened output

    Args:
        input: a 4D node.
        name: name scope of the layer.

    Returns:
        tensorflow tensor: a 2D node.
    """
    with tf.variable_scope(name) as scope:
        output = tf.nn.lrn(input)
    return output

def unflatten_layer (input, channels = 1, name = 'unflatten'):
    """
    This layer returns the unflattened output
    Args:
        input: a 2D node.
        chanels: How many channels are there in the image. (Default = ``1``)
        name: name scope of the layer.

    Returns:
        tensorflow tensor: a 4D node in (NHWC) format that is square in shape.
    """
    with tf.variable_scope(name) as scope:
        dim = int( np.sqrt( input.shape[1].value / channels ) ) 
        output = tf.reshape(input, [-1, dim, dim, channels])
    return output

def dropout_layer (input, prob, name ='dropout'):
    """
    This layer drops out nodes with the probability of 0.5
    During training time, run a probability of 0.5.
    During test time run a probability of 1.0. 
    To do this, ensure that the ``prob`` is a ``tf.placeholder``.
    You can supply this probability with ``feed_dict`` in trainer.

    Args:
        input: a 2D node.
        prob: Probability feeder.
        name: name scope of the layer.  

    Returns:
        tensorflow tensor: An output node           
    """ 
    with tf.variable_scope (name) as scope:
        output = tf.nn.dropout (input, prob)
    return output

if __name__ == '__main__':
    pass  