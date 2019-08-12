import tensorflow as tf
import numpy as np
from dnn_bl02_para  import *

#======================================================================================================#

class dnn_bl02_conf(object):

    def __init__(self, input_val, input_dim, mode):

        self.dnn_para  = dnn_bl02_para()
        self.input_val = input_val
        self.input_dim = input_dim
        self.mode      = mode

        ### ======== Layer 01: full connection
        with tf.device('/gpu:0'), tf.variable_scope("fully_layer01") as scope:
            self.output_layer01 = self.fully_layer(
                                                    self.input_val,
                                                    self.input_dim,

                                                    self.dnn_para.l01_fc, 

                                                    self.dnn_para.l01_is_act,
                                                    self.dnn_para.l01_act_func, 

                                                    self.dnn_para.l01_is_drop,
                                                    self.dnn_para.l01_drop_prob,

                                                    self.mode,
                                                    scope=scope
                                                    )
        ### ======== Layer 02: full connection
        with tf.device('/gpu:0'), tf.variable_scope("fully_layer02") as scope:
            self.output_layer02 = self.fully_layer(
                                                    self.output_layer01,
                                                    self.dnn_para.l01_fc,

                                                    self.dnn_para.l02_fc, 

                                                    self.dnn_para.l02_is_act,
                                                    self.dnn_para.l02_act_func, 

                                                    self.dnn_para.l02_is_drop,
                                                    self.dnn_para.l02_drop_prob,

                                                    self.mode,
                                                    scope=scope
                                                    )
        ### ======== Layer 03: full connection
        with tf.device('/gpu:0'), tf.variable_scope("fully_layer03") as scope:
            self.output_layer03 = self.fully_layer(
                                                    self.output_layer02,
                                                    self.dnn_para.l02_fc,

                                                    self.dnn_para.l03_fc, 

                                                    self.dnn_para.l03_is_act,
                                                    self.dnn_para.l03_act_func, 

                                                    self.dnn_para.l03_is_drop,
                                                    self.dnn_para.l03_drop_prob,

                                                    self.mode,
                                                    scope=scope
                                                    )
        ### ======== Layer 04: full connection
        with tf.device('/gpu:0'), tf.variable_scope("fully_layer04") as scope:
            self.output_layer04 = self.fully_layer(
                                                    self.output_layer03,
                                                    self.dnn_para.l03_fc,

                                                    self.dnn_para.l04_fc, 

                                                    self.dnn_para.l04_is_act,
                                                    self.dnn_para.l04_act_func, 

                                                    self.dnn_para.l04_is_drop,
                                                    self.dnn_para.l04_drop_prob,

                                                    self.mode,
                                                    scope=scope
                                                    )

 
            self.final_output = self.output_layer04

    ### 02/ FULL CONNECTTION  LAYER
    def fully_layer(
                     self, 
                     input_val, 
                     input_size, 
                     output_size, 
                     is_act,
                     act_func,
                     is_drop,
                     drop_prob, 
                     mode,
                     scope=None
                   ):

        with tf.variable_scope(scope or 'fully-layer') as scope:
            #initial parameter
            W    = tf.random_normal([input_size, output_size], stddev=0.1, dtype=tf.float32)
            bias = tf.random_normal([output_size], stddev=0.1, dtype=tf.float32)
            W    = tf.Variable(W)
            bias = tf.Variable(bias)

            #Dense 
            dense_output = tf.add(tf.matmul(input_val, W), bias)  

            #Active function
            if(is_act == True):
                if (act_func == 'RELU'):    
                    act_func_output = tf.nn.relu(dense_output)   
                elif (act_func == 'SOFTMAX'):
                    act_func_output  = tf.nn.softmax(dense_output)             
                elif (act_func == 'TANH'):
                    act_func_output  = tf.nn.tanh(dense_output)                 
            else:
                act_func_output = dense_output

            #Drop out
            if(is_drop == True):
                drop_output = tf.layers.dropout(
                                                act_func_output, 
                                                rate = drop_prob,
                                                training = mode,
                                                name = 'Dropout'
                                               )
            else:
                drop_output = act_func_output

            #Return 
            return drop_output

#===================================================================================
