# Some Global Defaults
C1 = 20     # Number of filters in first conv layer
C2 = 50     # Number of filters in second conv layer
D1 = 800    # Number of neurons in first dot-product layer
D2 = 800    # Number of neurons in second dot-product layer
C = 10      # Number of classes in the dataset to predict   
F1 = (5,5)  # Size of the filters in the first conv layer
F2 = (3,3)  # Size of the filters in the second conv layer
LR = 0.01   # Learning rate 
WEIGHT_DECAY_COEFF = 0.0001 # Co-Efficient for weight decay
L1_COEFF = 0.0001 # Co-Efficient for L1 Norm
MOMENTUM = 0.7 # Momentum rate 
OPTIMIZER = 'adam' # Optimizer (options include 'adam', 'rmsprop') Easy to upgrade if needed.
DROPOUT_PROBABILITY = 0.5 # Probability to dropout with.