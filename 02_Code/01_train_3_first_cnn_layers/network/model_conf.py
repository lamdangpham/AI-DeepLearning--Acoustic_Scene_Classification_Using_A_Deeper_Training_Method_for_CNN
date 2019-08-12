import tensorflow as tf
import numpy as np
import os

from model_para    import *
from cnn_bl_conf   import *
from dnn_bl01_conf import *
from dnn_bl02_conf import *
from add_bl_conf   import *


#======================================================================================================#

class model_conf(object):

    def __init__( self):

        self.model_para = model_para()

        # ============================== Fed Input
        #input data
        self.input_layer_val     = tf.placeholder(tf.float32, [None, self.model_para.n_freq, self.model_para.n_time, self.model_para.n_chan], name="input_layer_val")

        #expected class
        self.expected_classes    = tf.placeholder(tf.float32, [None, self.model_para.n_class], name="expected_classes")

        #run mode
        self.mode                = tf.placeholder(tf.bool, name="running_mode")

        #============================== NETWORK CONFIGURATION

        # Call Batchnorm
        with tf.device('/gpu:0'), tf.variable_scope("bn_01")as scope:
             self.input_layer_val_01 = tf.contrib.layers.batch_norm(self.input_layer_val, 
                                                                    is_training = self.mode, 
                                                                    decay = 0.9,
                                                                    zero_debias_moving_mean=True
                                                                   )

        # Call CNN and Get CNN output
        with tf.device('/gpu:0'), tf.variable_scope("cnn_01")as scope:
            self.cnn_ins_01 = cnn_bl_conf(self.input_layer_val_01, self.mode)

            self.cnn_ins_01_output = self.cnn_ins_01.final_output
            self.cnn_ins_01_mid_01 = self.cnn_ins_01.mid_layer01 #32
            self.cnn_ins_01_mid_02 = self.cnn_ins_01.mid_layer02 #64
            self.cnn_ins_01_mid_03 = self.cnn_ins_01.mid_layer03 #128


        #Call branch DNN  
        with tf.device('/gpu:0'), tf.variable_scope("dnn_02_01")as scope:
            self.dnn_bl02_ins_01 = dnn_bl02_conf(self.cnn_ins_01_mid_01, 32, self.mode)
            self.br01            = self.dnn_bl02_ins_01.final_output
        with tf.device('/gpu:0'), tf.variable_scope("dnn_02_02")as scope:
            self.dnn_bl02_ins_02 = dnn_bl02_conf(self.cnn_ins_01_mid_02, 64, self.mode)
            self.br02            = self.dnn_bl02_ins_02.final_output
        with tf.device('/gpu:0'), tf.variable_scope("dnn_02_03")as scope:
            self.dnn_bl02_ins_03 = dnn_bl02_conf(self.cnn_ins_01_mid_03, 128, self.mode)
            self.br03            = self.dnn_bl02_ins_03.final_output

    
        # Call main DNN and Get main DNN output
        with tf.device('/gpu:0'), tf.variable_scope("dnn_01")as scope:
            self.dnn_bl01_ins_01 = dnn_bl01_conf(self.cnn_ins_01_output, 256, self.mode)

            self.output_layer      = self.dnn_bl01_ins_01.final_output
            self.prob_output_layer = tf.nn.softmax(self.output_layer)
            self.wanted_data       = self.cnn_ins_01_output

        ### ======================================== LOSS FUNCTION AND ACCURACY =========================
        ### loss function
        with tf.device('/gpu:0'), tf.variable_scope("loss") as scope:
  
            # l2 loss  
            l2_loss = self.model_para.l2_lamda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

            # main loss
            losses      = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.output_layer)
            losses_br01 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.br01)
            losses_br02 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.br02)
            losses_br03 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.br03)

            # final loss
            #self.loss = tf.reduce_mean(losses) + l2_loss + losses_br01 
            #self.loss = tf.reduce_mean(losses) + l2_loss + losses_br01 + losses_br02 
            self.loss = tf.reduce_mean(losses) + l2_loss + losses_br01 + losses_br02 + losses_br03

        ### Calculate Accuracy  
        with tf.device('/gpu:0'), tf.name_scope("accuracy") as scope:
            self.correct_prediction = tf.equal(tf.argmax(self.output_layer,1),    tf.argmax(self.expected_classes,1))
            self.accuracy           = tf.reduce_mean(tf.cast(self.correct_prediction,"float", name="accuracy" ))
