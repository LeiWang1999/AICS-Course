import tensorflow as tf
from math import sqrt
from .third_party import put_kernels_on_grid

def initializer(shape, name = 'xavier'):
    """
    A method that returns random numbers for Xavier initialization.

    Args:
        shape: shape of the initializer.
        name: Name for the scope of the initializer

    Returns:
        float: random numbers from tensorflow random_normal

    """
    with tf.variable_scope(name) as scope:
        stddev = 1.0 / tf.sqrt(float(shape[0]), name = 'stddev')
        inits = tf.truncated_normal(shape=shape, stddev=stddev, name = 'xavier_init')
    return inits

def nhwc2hwnc (nhwc, name = 'nhwc2hwnc'):
    """
    This method reshapes (NHWC) 4D bock to (HWNC) 4D block

    Args:
        nhwc: 4D block in (NHWC) format

    Returns:
        tensorflow tensor: 4D block in (HWNC) format
    """    
    with tf.variable_scope(name) as scope:
        out = tf.transpose(nhwc, [1,2,0,3])
    return out

def nhwc2hwcn (nhwc, name = 'nhwc2hwcn'):
    """
    This method reshapes (NHWC) 4D bock to (HWCN) 4D block

    Args:
        nhwc: 4D block in (NHWC) format

    Returns:
        tensorflow tensor: 4D block in (HWCN) format
    """    
    with tf.variable_scope(name) as scope:
        out = tf.transpose(nhwc, [1,2,3,0])
    return out

def visualize_filters (filters, name = 'conv_filters'):
    """
    This method is a wrapper to ``put_kernels_on_grid``. This adds the grid to image summaries.

    Args:
        tensorflow tensor: A 4D block in (HWNC) format.
    """
    grid = put_kernels_on_grid (filters, name = 'visualizer_' + name) 
    tf.summary.image(name, grid, max_outputs = 1)

def visualize_images (images, name = 'images', num_images = 6):
    """
    This method sets up summaries for images.

    Args:
        images: a 4D block in (NHWC) format.
        num_images: Number of images to display

    Todo:
        I want this to display images in a grid rather than just display using 
        tensorboard's ugly system. This method should be a wrapper that converts 
        images in (NHWC) format to (HWNC) format and makes a grid of the images.
        
        Perhaps a code like this:

        ```            
            images = images [0:num_images-1]
            images = nhwc2hwcn(images, name = 'nhwc2hwcn' + name)
            visualize_filters(images, name)        
    """
    tf.summary.image(name, images, max_outputs = num_images)