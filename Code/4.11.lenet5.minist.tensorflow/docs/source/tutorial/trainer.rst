Trainer
=======

.. todo::

    This part of the tutorial is currently being done.

The trainer is perhaps the module that is most unique to tensorflow and is most different from theano.
Tensroflow uses :meth:`tf.Session` to parse computational graphs unlike in theano where we'd use 
the :meth:`theano.function` methods. 
For a detailed tutorial on how tensorflow processes and runs graphs, refer 
`this page <https://www.tensorflow.org/api_guides/python/client>`_.

The :class:`lenet.trainer.trainer` class takes as input an object of :class:`lenet.network.lenet5`
and :class:`lenet.dataset.mnist`. 
After adding them as attributes, it then initializes a new tensorflow session to run the 
computational graph and initializes all the variables in the graph.

.. code-block:: python

    self.network = network
    self.dataset = dataset 
    self.session = tf.InteractiveSession()        
    tf.global_variables_initializer().run()

The initializer class also calls the :meth:`lenet.trainer.trainer.summaries` method that initializes 
the summary writer (:meth:`tf.summary.FileWriter`) so that any processing on this computational graph
could be monitored at tensorboard. 

.. code-block:: python

    self.summary = tf.summary.merge_all()
    self.tensorboard = tf.summary.FileWriter("tensorboard")
    self.tensorboard.add_graph(self.session.graph)

The mnist example from :meth:`tf.examples.tutorials.mnist.input_data` that we use here as ``self.dataset`` is 
written in such a way that given a ``mini_batch_size``, we can easily query and retrieve the next batch 
as follows:

.. code-block:: python

    x, y = self.dataset.train.next_batch(mini_batch_size)

While in theano, we would use the :meth:`theano.function` method to produce the function to run back prop updates,
here we can use the minimizer that we created in :class:`lenet.network.lenet5` (``self.network.back_prop`` in 
:class:`lenet.trainer.trainer`) to run one update step. We also want to collect (*fetch* is the tensorflow terminology)
``self.network.obj`` and ``self.network.cost`` (see definitions at :class:`lenet.network.lenet5`) to be able to 
monitor the network training. 
All this can be done using the following code:

.. code-block:: python

    _, obj, cost  = self.session.run(  
                        fetches = [self.network.back_prop, self.network.obj, self.network.cost], \
                        feed_dict = {self.network.images:x, self.network.labels:y, \
                                    self.network.dropout_prob: DROPOUT_PROBABILITY})

This is similar to how we'd run a :meth:`theano.function`. The ``givens`` operation which is used in theano 
to feed values to placeholders is now supplied here using ``feed_dict`` which takes in a dictionary, whose
key, value pair is a node and its initialization value. Here we assign to ``self.network.images`` the 
images we just retrieved, to ``self.network.labels`` the ``y`` we just queried and to ``self.network.dropout_prob``
which was the node controlling the dropout Bernoulli probability, the gloabl defined dropout. 
We use this value of dropout, since this does back prop.
If we were just feeding forward without updating the weights (such as during inference or test) we would not 
use this probability, instead we would use,

.. code-block:: python 

    acc = self.session.run(self.network.accuracy,\
                            feed_dict = { self.network.images: x,
                                            self.network.labels: y,
                                            self.network.dropout_prob: 1.0} )

as was used in the :meth:`lenet.trainer.trainer.accuracy`. 
The same :meth:`lenet.trainer.trainer.accuracy` method with different placeholders 
could be used for testing and training accuracies.

.. code-block:: python 

    # Testing
    x = self.dataset.test.images
    y = self.dataset.test.labels
    acc = self.accuracy (images =x, labels = y)  

    # Training              
    x, y = self.dataset.train.next_batch(mini_batch_size)
    acc = self.accuracy (images =x, labels = y)       

After a desired number of iterations, we might want to update the tensorboard summaries or print out 
a cost to use for reference on how well we are training.
We can use ``self.session``, which is the same session previously used, to write out all summaries.
This run call to session will write out everything we have added to the summaries along the way of 
building the network itself using our ``self.tensorboard`` writer.

.. code-block:: python

    x = self.dataset.test.images
    y = self.dataset.test.labels
    s = self.session.run(self.summary, feed_dict = {self.network.images: x,
                                                    self.network.labels: y,
                                                    self.network.dropout_prob: 1.0})
    self.tensorboard.add_summary(s, iter)

The last thing we have to define is the the :meth:`lenet.trainer.trainer.train` method. 
This method will run the training loops for the network that we have definied, taking in input arguments
``iter= 10000``,  ``mini_batch_size = 500``, ``update_after_iter = 1000``,
``summarize = True``, with obviously named variables. 

The trainer loop can be coded as:

.. code-block:: python

    # Iterate over iter
    for it in range(iter):            
        obj, cost = self.bp_step(mini_batch_size)  # Run a step of back prop minimizer         
        if it % update_after_iter == 0:            # Check if it is time to flush out summaries.    
            train_acc = self.training_accuracy(mini_batch_size = 50000)
            acc = self.test()                      # Measure training and testing accuracies.
            print(  " Iter " + str(it) +           # Print them on terminal.
                    " Objective " + str(obj) +
                    " Cost " + str(cost) + 
                    " Test Accuracy " + str(acc) +
                    " Training Accuracy " + str(train_acc) 
                    )                   
            if summarize is True:                  # Write summaries to tensorflow
                self.write_summary(iter = it, mini_batch_size = mini_batch_size)

The above code essentially iterates over ``iter`` supplied to the method and runs one step of 
``self.network.back_prop`` method, which we cooked in :meth:`lenet.network.lenet5.cook`.
If it was time to flush out summaries it does so.
Finally, once the training is complete, we can call the :meth:`lenet.trainer.trainer.test` method to 
produce testing accuracies. 

.. code-block:: python

    acc = self.test()
    print ("Final Test Accuracy: " + str(acc))   


Since everything else, including the first layer filters and confusion matrices were all stored in 
summaries, they would have been adequately flushed out.

The trainer class documentation can be found in :ref:`trainer`. 