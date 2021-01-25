Introduction 
============

The features that were classic to computer vision were designed by hand. 
Some, like PCA were designed with a particular objective, such as creating representations on basis where the variance was large. 
These features were, for the most part, general enough that they were used for a variety of tasks. 
Features like HOG :cite:`dalal2005histograms` and SIFT :cite:`lowe1999object` for instance, have been used for tasks including video activity recognition 
:cite:`wu2007scalable`, :cite:`sun2010activity`, :cite:`oreifej2013hon4d`, vehicle detection :cite:`kembhavi2011vehicle`, :cite:`sun2006road`, object tracking
:cite:`zhou2009object`, :cite:`zhang2013structure`, pedestrian detection :cite:`dollar2012pedestrian` and face detection :cite:`zhu2006fast`,
:cite:`luo2007person`, :cite:`ke2004pca`, just to list a few. 
While this school of thought continue to be quite popular and some of these features have standardized implementations that are available for most researchers 
to plug and play for their tasks, they were not task-specific. 
These were designed to be useful feature representations of images that are capable of providing cues about certain aspects of images. 
HOG and SIFT for instance, provided shape-related information and were therefore used in tasks involving shape and structure. 
Features like color correlogram :cite:`venkatesan2012classification` provided cues on color transitions and were therefore used in medical
images and other problems where, shape was not necessarily an informative feature.
In this tutorial we will study a popular technique used in *learning to create good features* and task-specific feature extractors. 
These machine learn to extract useful feature representations of images using the data for a particular task.

Multi-layer neural networks have long since been viewed as a means of extracting hierarchical task-specific features. 
Ever since the early works of Rumelhart et al., :cite:`rumelhart1985learning` it was recognized that representations learnt using 
back-propagation had the potential to learn fine-tuned features that were task-specific. 
Until the onset of this decade, these methods were severely handicapped by a dearth of large-scale data and large-scale parallel
compute hardware to be leveraged sufficiently. 
This, in part, directed the creativity of computer vision scientists to develop the aforementioned general-purpose
feature representations. 
We now have access to datasets that are large enough and GPUs that are capable of large-scale parallel computations. 
This has allowed an explosion in neural image features and their usage. 
In the next sections we will study some of these techniques.

An artificial neural network is a network of computational neurons that are connected in a directed acyclic graph. 
There are several types of neural networks. 
While dealing with images, we are mostly concerned with the use the convolutional neural network (CNN).
Each neuron accepts a number of inputs and produces one output, which can further be supplied to many other neurons. 
A typical function of a computational neuron is to weight all the inputs, sum all the weighted inputs and generate an output depending on the strength of 
the summed weighted inputs. 
Neurons are organized in groups, where each group typically receives input from the same sources. 
These groups are called as layers. 
Layers come in three varieties, each characterized by its own type of a neuron. 
They are, the dot-product or the fully-connected layer, the convolutional layer and the pooling layer. 