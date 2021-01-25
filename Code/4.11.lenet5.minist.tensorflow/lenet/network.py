import tensorflow as tf
import numpy as np
from .layers import *
from .support import visualize_images
from .global_definitions import *

def apply_gradient_descent(var_list, obj):
    """
    Sets up the gradient descent optimizer

    Args:
        var_list: List of variables to optimizer over.        
        obj: Node of the objective to minimize
    Notes:
        learning_rate: What learning rate to run with. (Default = ``0.01``) Set with ``LR``
    """
    back_prop = tf.train.GradientDescentOptimizer(
                                            learning_rate = LR,
                                            name = 'gradient_descent' ).minimize(loss = obj, \
                                                    var_list = var_list ) 
    return back_prop

def apply_adam (var_list, obj, learning_rate = 1e-4):
    """
    Sets up the ADAM optimizer

    Args:
        var_list: List of variables to optimizer over.
        obj: Node of the objective to minimize        
    
    Notes:
        learning_rate: What learning rate to run with. (Default = ``0.01``) Set with ``LR``
    """      
    back_prop = tf.train.AdamOptimizer(
                                        learning_rate = LR,
                                        name = 'adam' ).minimize(loss = obj, \
                                            var_list = var_list) 
    return back_prop                                                               

def apply_rmsprop( var_list, obj ):
    """
    Sets up the RMS Prop optimizer

    Args:
        var_list: List of variables to optimizer over.
        obj: Node of the objective to minimize        

    Notes:
        * learning_rate: What learning rate to run with. (Default = ``0.001``). Set  ``LR``
        * momentum: What is the weight for momentum to run with. (Default = ``0.7``). Set ``MOMENTUM``
        * decay: What rate should learning rate decay. (Default = ``0.95``). Set ``DECAY``            
    """    
    back_prop = tf.train.RMSPropOptimizer(
                                        learning_rate = LR,
                                        decay = DECAY,
                                        momentum = MOMENTUM,
                                        name = 'rmsprop' ).minimize(loss = obj, \
                                        var_list = var_list) 
    return back_prop

def apply_weight_decay (var_list, name = 'weight_decay'):
    """
    This method applies L2 Regularization to all weights and adds it to the ``objectives`` 
    collection. 
    
    Args:
        name: For the tensorflow scope.
        var_list: List of variables to apply.
    
    Notes:
        What is the co-efficient of the L2 weight? Set ``WEIGHT_DECAY_COEFF``.( Default = 0.0001 )
    """                              
    for param in var_list:
        norm = WEIGHT_DECAY_COEFF * tf.nn.l2_loss(param)
        tf.summary.scalar('l2_' + param.name, norm)                  
        tf.add_to_collection('objectives', norm)

def apply_l1 ( var_list, name = 'l1'):
    """
    This method applies L1 Regularization to all weights and adds it to the ``objectives`` 
    collection. 
    
    Args:
        var_list: List of variables to apply l1
        name: For the tensorflow scope.
    
    Notes:
        What is the co-efficient of the L1 weight? Set ``L1_COEFF``.( Default = 0.0001 )
    """                              
    for param in var_list:
        norm = L1_COEFF * tf.reduce_sum(tf.abs(param, name = 'abs'), name = 'l1')
        tf.summary.scalar('l1_' + param.name, norm)                  
        tf.add_to_collection( 'objectives', norm)

def process_params(params):
    """
    This method adds the params to two collections.
    The first element is added to ``regularizer_worthy_params``.
    The first and second elements are is added to ``trainable_parmas``.

    Args:
        params: List of two.
    """
    tf.add_to_collection( 'trainable_params', params[0])
    tf.add_to_collection( 'trainable_params', params[1])         
    tf.add_to_collection('regularizer_worthy_params', params[0]) 

def apply_regularizer ( var_list):
    """
    This method applyies Regularization to all weights and adds it to the ``objectives`` 
    collection. 
    
    Args:
        var_list: List of variables to apply l1
    
    Notes:
        What is the co-efficient of the L1 weight? Set ``L1_COEFF``.( Default = 0.0001 )
    """       
    with tf.variable_scope( 'weight-decay') as scope:
        if WEIGHT_DECAY_COEFF > 0:
            apply_weight_decay(name = 'weight_decay', var_list = var_list )

    with tf.variable_scope( 'l1-regularization') as scope:
        if L1_COEFF > 0:
            apply_l1(name = 'weight_decay',  var_list = var_list)


class lenet5(object):
    """
    Definition of the lenet class of networks.

    Notes:
        *   Produces the lenet model and returns the weights. A typical lenet has 
            two convolutional layers with filters sizes ``5X5`` and ``3X3``. These
            are followed by two fully-connected layers and a softmax layer. This 
            network model, reproduces this network to be trained on MNIST images
            of size ``28X28``.     
        *   Most of the important parameters are stored in :mod:`global_definitions` 
            in the file ``global_definitions.py``.

    Args:
        images: Placeholder for images

    Attributes:        
        images: This is the placeholder for images. This needs to be fed in from :class:`lenet.dataset.mnist``.
        dropout_prob: This is also a placeholder for dropout probability. This needs to be fed in.    
        logits: Output node of the softmax layer, before softmax. This is an output from a 
                :meth:`lenet.layers.dot_product_layer`.
        inference: Output node of the softmax layer that produces inference.
        predictions: Its a predictions node which is :meth:`tf.nn.argmax` of ``inference``. 
        back_prop: Backprop is an optimizer. This is a node that will be used by a :class:`lenet.trainer.trainer` later.
        obj: Is a cumulative objective tensor. This produces the total summer objective in a node.
        cost: Cost of the back prop error alone. 
        labels: Placeholder for labels, needs to be fed in. This is added fed in from the dataset class.
        accuracy: Tensor for accuracy. This is a node that measures the accuracy for the mini batch.

    """
    def __init__ (  self,
                    images ):
        """
        Class constructor. Creates the model and allthe connections. 
        """
        self.images = images
        # Unflatten Layer
        images_square = unflatten_layer ( self.images )
        visualize_images(images_square)

        # Conv Layer 1
        conv1_out, params =  conv_2d_layer (    input = images_square,
                                                neurons = C1,
                                                filter_size = F1,
                                                name = 'conv_1',
                                                visualize = True )
        process_params(params)
        pool1_out = max_pool_2d_layer ( input = conv1_out, name = 'pool_1')
        lrn1_out = local_response_normalization_layer (pool1_out, name = 'lrn_1' )

        # Conv Layer 2
        conv2_out, params =  conv_2d_layer (    input = lrn1_out,
                                                neurons = C2,
                                                filter_size = F2,
                                                name = 'conv_2' )
        process_params(params)
        
        pool2_out = max_pool_2d_layer ( input = conv2_out, name = 'pool_2')
        lrn2_out = local_response_normalization_layer (pool2_out, name = 'lrn_2' )

        flattened = flatten_layer(lrn2_out)

        # Placeholder probability for dropouts.
        self.dropout_prob = tf.placeholder(tf.float32,
                                            name = 'dropout_probability')

        # Dropout Layer 1 
        flattened_dropout = dropout_layer ( input = flattened,
                                          prob = self.dropout_prob,
                                          name = 'dropout_1')                                          

        # Dot Product Layer 1
        fc1_out, params = dot_product_layer  (  input = flattened_dropout,
                                                neurons = D1,
                                                name = 'dot_1')
        process_params(params)

        # Dropout Layer 2 
        fc1_out_dropout = dropout_layer ( input = fc1_out,
                                          prob = self.dropout_prob,
                                          name = 'dropout_2')
        # Dot Product Layer 2
        fc2_out, params = dot_product_layer  (  input = fc1_out_dropout, 
                                                neurons = D2,
                                                name = 'dot_2')
        process_params(params)

        # Dropout Layer 3 
        fc2_out_dropout = dropout_layer ( input = fc2_out,
                                          prob = self.dropout_prob,
                                          name = 'dropout_3')

        # Logits layer
        self.logits, params = dot_product_layer  (  input = fc2_out_dropout,
                                                    neurons = C,
                                                    activation = 'identity',
                                                    name = 'logits_layer')
        process_params(params)

        # Softmax layer
        self.inference, self.predictions = softmax_layer (  input = self.logits,
                                                            name = 'softmax_layer' )                                                    
                
    def cook(self, labels):
        """
        Prepares the network for training

        Args:
            labels: placeholder for labels

        Notes:
            *   Each optimizer has a lot parameters that, if you want to change, modify in the code
                directly. Most do not take in inputs and runs. Some parameters such as learning rates 
                play a significant role in learning and are good choices to experiment with.
            *   what optimizer to run with. (Default = ``sgd``), other options include
                'rmsprop' and 'adam'. Set ``OPTIMIZER``
        """    
        with tf.variable_scope('objective') as scope:
            self.labels = labels
            with tf.variable_scope('cross-entropy') as scope:
                loss = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits ( 
                                                     labels = self.labels,
                                                     logits = self.logits)
                                                )
                self.cost = loss
                tf.add_to_collection('objectives', loss ) 
                tf.summary.scalar('cost', loss)  

            apply_regularizer (var_list = tf.get_collection( 'regularizer_worthy_params') )
            self.obj = tf.add_n(tf.get_collection('objectives'), name='objective')
            tf.summary.scalar('obj', self.obj)  


        with tf.variable_scope('train') as scope:
            # Change (supply as arguments) parameters here directly in the code.
            if OPTIMIZER == 'sgd':                                                                              
                self.back_prop = apply_gradient_descent(var_list = tf.get_collection( \
                                                            'trainable_params'),
                                                            obj = self.obj )
            elif OPTIMIZER == 'rmsprop':
                self.back_prop = apply_rmsprop(var_list = tf.get_collection( \
                                                            'trainable_params') ,
                                                            obj = self.obj)
            elif OPTIMIZER == 'adam':
                self.back_prop = apply_adam (var_list = tf.get_collection( \
                                                            'trainable_params') ,
                                                            obj = self.obj )
            else:
                raise Error('Invalid entry to optimizer')
                

        with tf.variable_scope('test') as scope:                                                
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1), \
                                                        name = 'correct_predictions')
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32) , name ='accuracy')                                     
            tf.summary.scalar('accuracy', self.accuracy) 
     
            with tf.variable_scope("confusion"):
                confusion = tf.confusion_matrix(tf.argmax(self.labels,1), self.predictions,
                                                num_classes=C,
                                                name='confusion')
                confusion_image = tf.reshape( tf.cast( confusion, tf.float32),[1, C, C, 1])
                tf.summary.image('confusion',confusion_image)                       