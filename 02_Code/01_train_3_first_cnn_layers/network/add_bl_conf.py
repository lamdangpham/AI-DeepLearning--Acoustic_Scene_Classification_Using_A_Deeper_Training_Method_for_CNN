import tensorflow as tf
import numpy as np
import os

#======================================================================================================#

class add_bl_conf(object):

    def __init__( self, input01, input02, input03):

        self.cnn_output_01 = input01
        self.cnn_output_02 = input02
        self.cnn_output_03 = input03

        with tf.device('/gpu:0'), tf.variable_scope("Addition") as scope:

             [_,w_size] = self.cnn_output_01.get_shape()  

             W1    = tf.random_normal([int(w_size)], stddev=0.1, dtype=tf.float32)
             W1    = tf.Variable(W1)
             W2    = tf.random_normal([int(w_size)], stddev=0.1, dtype=tf.float32)
             W2    = tf.Variable(W2)
             W3    = tf.random_normal([int(w_size)], stddev=0.1, dtype=tf.float32)
             W3    = tf.Variable(W3)

             self.cnn_output_01_w = self.cnn_output_01 * W1  #tf.multiply(self.cnn_output_01, W1)
             self.cnn_output_02_w = self.cnn_output_02 * W2
             self.cnn_output_03_w = self.cnn_output_03 * W3
  
             #Add
             self.cnn_output_w     = self.cnn_output_01_w +  self.cnn_output_02_w + self.cnn_output_03_w
             self.cnn_output_w_act = tf.nn.relu(self.cnn_output_w, name="RELU")

             #Concat
             #self.cnn_output_w     = tf.concat([self.cnn_output_01_w, self.cnn_output_02_w, self.cnn_output_03_w], 0)
             #self.cnn_output_w_act = tf.nn.relu(self.cnn_output_w, name="RELU")

             #print self.cnn_output_01_w.get_shape()
             #print self.cnn_output_02_w.get_shape()
             #print self.cnn_output_03_w.get_shape()
             #print self.cnn_output_w.get_shape()


        # Plattenning
        with tf.device('/gpu:0'), tf.variable_scope("platterning-l04") as scope:
             [_,col] = self.cnn_output_w_act.get_shape()   #nx1x256
             #print col

             self.cnn_output_w_act_dim = int(col)

             self.cnn_output_w_act_flat  = tf.reshape(self.cnn_output_w_act, [-1, self.cnn_output_w_act_dim])
             #print self.cnn_output_w_act_flat.get_shape()


             self.final_output = self.cnn_output_w_act_flat
             self.final_output_dim = self.cnn_output_w_act_dim
