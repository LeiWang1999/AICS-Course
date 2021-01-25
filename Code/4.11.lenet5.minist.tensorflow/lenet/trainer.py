import tensorflow as tf
from .global_definitions import DROPOUT_PROBABILITY

class trainer(object):
    """
    Trainer for networks

    Args:
        network: A network class object
        dataset: A tensorflow dataset object

    Attributes:        
        network: This is the network we initialized with. We pass this as an argument and we add it 
                to the current trainer class.
        dataset: This is also the initializer. It comes from the :class:`lenet.dataset.mnist` module.
        session: This is a session created with trainer. This session is used for training.
        tensorboard: Is a summary writer tool. This writes things into the tensorboard that is 
                    then setup on the tensorboard server. At the end of the trainer, it closes 
                    this tensorboard.
        
    """
    def __init__ (self, network, dataset):
        """
        Class constructor
        """
        self.network = network
        self.dataset = dataset 
        self.session = tf.InteractiveSession()        
        tf.global_variables_initializer().run()
        self.summaries()

    def bp_step(self, mini_batch_size):
        """
        Sample a minibatch of data and run one step of BP.

        Args:
            mini_batch_size: Integer
        
        Returns: 
            tuple of tensors: total objective and cost of that step
        """
        x, y = self.dataset.train.next_batch(mini_batch_size)
        _, obj, cost  = self.session.run(  
                           fetches = [self.network.back_prop, self.network.obj, self.network.cost], \
                           feed_dict = {self.network.images:x, self.network.labels:y, \
                                        self.network.dropout_prob: DROPOUT_PROBABILITY})
        return (obj, cost)

    def accuracy (self, images, labels):
        """
        Return accuracy

        Args:
            images: images
            labels: labels

        Returns:
            float: accuracy            
        """
        acc = self.session.run(self.network.accuracy,\
                               feed_dict = { self.network.images: images,
                                             self.network.labels: labels,
                                             self.network.dropout_prob: 1.0} )
        return acc

    def summaries(self, name = "tensorboard"):
        """
        Just creates a summary merge bufer

        Args:
            name: a name for the tensorboard directory
        """
        self.summary = tf.summary.merge_all()
        self.tensorboard = tf.summary.FileWriter("tensorboard")
        self.tensorboard.add_graph(self.session.graph)


    def test (self):
        """
        Run validation of the model  

        Returns:
            float: accuracy                     
        """
        x = self.dataset.test.images
        y = self.dataset.test.labels
        acc = self.accuracy (images =x, labels = y)                
        return acc

    def training_accuracy (self, mini_batch_size = 500):
        """
        Run validation of the model on training set   

        Args:
            mini_batch_size: Number of samples in a mini batch 
            
        Returns:
            float: accuracy                      
        """
        x, y = self.dataset.train.next_batch(mini_batch_size)
        acc = self.accuracy (images =x, labels = y)                
        return acc

    def write_summary (self, iter = 0, mini_batch_size = 500):
        """
        This method updates the summaries
        
        Args:
            iter: iteration number to index values with.
            mini_batch_size: Mini batch to evaluate on.
        """
        x = self.dataset.test.images
        y = self.dataset.test.labels
        s = self.session.run(self.summary, feed_dict = {self.network.images: x,
                                                        self.network.labels: y,
                                                        self.network.dropout_prob: 1.0})
        self.tensorboard.add_summary(s, iter)

    def train ( self, 
                iter= 10000, 
                mini_batch_size = 500, 
                update_after_iter = 1000, 
                training_accuracy = False,
                summarize = True):
        """
        Run backprop for ``iter`` iterations

        Args:   
            iter: number of iterations to run
            mini_batch_size: Size of the mini batch to process with
            training_accuracy: if ``True``, will calculate accuracy on training data also.
            update_after_iter: This is the iteration for validation
            summarize: Tensorboard operation
        """
        for it in range(iter):            
            obj, cost = self.bp_step(mini_batch_size)            
            if it % update_after_iter == 0:           
                train_acc = self.training_accuracy(mini_batch_size = 50000)
                acc = self.test()
                print(  " Iter " + str(it) +
                        " Objective " + str(obj) +
                        " Cost " + str(cost) + 
                        " Test Accuracy " + str(acc) +
                        " Training Accuracy " + str(train_acc) 
                        )                   
                if summarize is True:
                    self.write_summary(iter = it, mini_batch_size = mini_batch_size)
        acc = self.test()
        print ("Final Test Accuracy: " + str(acc))            