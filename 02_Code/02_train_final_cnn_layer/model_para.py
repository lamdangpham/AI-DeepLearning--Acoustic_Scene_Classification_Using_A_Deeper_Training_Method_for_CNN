
#import tensorflow as tf
import numpy as np
import os

class model_para(object):
    """
    define a class to store parameters,
    the input should be feature matrix of training and testing
    """

    def __init__(self):

        #======================= Trainging parameters
        self.n_class            = 15  # Final output classes TODO
        self.n_output           = 15  # Final output   TODO
        self.l2_lamda           = 0.0001  # lamda prarameter

        #========================  Input parameters
        self.input_layer_dim    = 256  
  
        #========================  CNN structure parameters
        #=======Layer 01: full connection
        self.l01_fc             = 512 # node number of first full-connected layer 
        self.l01_is_act         = True
        self.l01_act_func       = 'RELU'
        self.l01_is_drop        = False
        self.l01_drop_prob      = 1

        #=======Layer 02: full connection
        self.l02_fc             = 1024 # node number of first full-connected layer 
        self.l02_is_act         = True
        self.l02_act_func       = 'RELU'
        self.l02_is_drop        = False
        self.l02_drop_prob      = 1

        #=======Layer 03: full connection
        self.l03_fc             = 1024  # node number of first full-connected layer 
        self.l03_is_act         = True
        self.l03_act_func       = 'RELU'
        self.l03_is_drop        = False
        self.l03_drop_prob      = 1

        #=======Layer 04: Final layer
        self.l04_fc             = 15   # output node number = class numbe
        self.l04_is_act         = False
        self.l04_act_func       = 'RELU'
        self.l04_is_drop        = False
        self.l04_drop_prob      = 1


