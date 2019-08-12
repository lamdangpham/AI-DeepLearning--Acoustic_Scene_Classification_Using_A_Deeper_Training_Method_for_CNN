import tensorflow as tf
import numpy as np
import os
from model_para  import model_para

#======================================================================================================#

class model_conf(object):

    def __init__( self):

        self.model_para = model_para()
        
        # These arguments are transfered from 'step02...' file
        self.input_layer_val   = tf.placeholder(tf.float32, [None, self.model_para.input_layer_dim], name="input_layer_val")
        self.expected_classes  = tf.placeholder(tf.float32, [None, self.model_para.n_class],         name="expected_classes")
        self.mode              = tf.placeholder(tf.bool, name="running_mode")


        ### ======== Layer 01: full connection
        with tf.device('/gpu:0'), tf.variable_scope("fully_layer01") as scope:
            self.output_layer01 = self.fully_layer(
                                                    self.input_layer_val,
                                                    self.model_para.input_layer_dim,

                                                    self.model_para.l01_fc, 

                                                    self.model_para.l01_is_act,
                                                    self.model_para.l01_act_func, 

                                                    self.model_para.l01_is_drop,
                                                    self.model_para.l01_drop_prob,
                                                    self.mode,
                                                    scope=scope
                                                    )
        ### ======== Layer 02: full connection
        with tf.device('/gpu:0'), tf.variable_scope("fully_layer02") as scope:
            self.output_layer02 = self.fully_layer(
                                                    self.output_layer01,
                                                    self.model_para.l01_fc,

                                                    self.model_para.l02_fc, 

                                                    self.model_para.l02_is_act,
                                                    self.model_para.l02_act_func, 

                                                    self.model_para.l02_is_drop,
                                                    self.model_para.l02_drop_prob,
                                                    self.mode,
                                                    scope=scope
                                                    )
        ### ======== Layer 03: full connection
        with tf.device('/gpu:0'), tf.variable_scope("fully_layer03") as scope:
            self.output_layer03 = self.fully_layer(
                                                    self.output_layer02,
                                                    self.model_para.l02_fc,

                                                    self.model_para.l03_fc, 

                                                    self.model_para.l03_is_act,
                                                    self.model_para.l03_act_func, 

                                                    self.model_para.l03_is_drop,
                                                    self.model_para.l03_drop_prob,
                                                    self.mode,
                                                    scope=scope
                                                    )

        ### ======== Layer 04: full connection
        with tf.device('/gpu:0'), tf.variable_scope("fully_layer04") as scope:
            self.output_layer04 = self.fully_layer(
                                                    self.output_layer03,
                                                    self.model_para.l03_fc,

                                                    self.model_para.l04_fc, 

                                                    self.model_para.l04_is_act,
                                                    self.model_para.l04_act_func, 

                                                    self.model_para.l04_is_drop,
                                                    self.model_para.l04_drop_prob,
                                                    self.mode,
                                                    scope=scope
                                                    )
 
            self.output_layer = self.output_layer04
            self.prob_output_layer = tf.nn.softmax(self.output_layer)

            #print self.output_layer04.get_shape()           #n x nClassa
            #exit()

        ### ======================================== LOSS FUNCTION AND ACCURACY =========================
        ### loss function
        with tf.device('/gpu:0'), tf.variable_scope("loss") as scope:
  
            # l2 loss  
            l2_loss = self.model_para.l2_lamda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

            # main loss
            losses  = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.output_layer)
 
            # final loss
            self.loss = tf.reduce_mean(losses) + l2_loss    #reduce_sum or reduce_mean
            #self.loss = tf.reduce_mean(losses)             

        ### Calculate Accuracy  
        with tf.device('/gpu:0'), tf.name_scope("accuracy") as scope:
            self.correct_prediction = tf.equal(tf.argmax(self.output_layer,1), tf.argmax(self.expected_classes,1))
            self.accuracy      = tf.reduce_mean(tf.cast(self.correct_prediction,"float", name="accuracy" ))


###==================================================== OTHER FUNCTION ============================
    ### 01/ FULL CONNECTTION  LAYER
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
