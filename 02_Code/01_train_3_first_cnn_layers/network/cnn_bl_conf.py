import tensorflow as tf
import numpy as np
import os
from cnn_bl_para  import *

#======================================================================================================#

class cnn_bl_conf(object):

    def __init__(self, input_layer_val, mode):

        self.cnn_para        = cnn_bl_para()
        self.input_layer_val = input_layer_val
        self.mode            = mode
        
        ### ======== LAYER 01
        with tf.device('/gpu:0'), tf.variable_scope("conv01")as scope:
             [self.output_layer01, self.mid_layer01] = self.conv_layer(
                                                   self.input_layer_val,

                                                   self.cnn_para.l01_filter_height,
                                                   self.cnn_para.l01_filter_width,
                                                   self.cnn_para.l01_pre_filter_num,
                                                   self.cnn_para.l01_filter_num,
                                                   self.cnn_para.l01_conv_padding,
                                                   self.cnn_para.l01_conv_stride,

                                                   self.cnn_para.l01_is_norm,

                                                   self.cnn_para.l01_conv_act_func,

                                                   self.cnn_para.l01_is_pool,
                                                   self.cnn_para.l01_pool_type,
                                                   self.cnn_para.l01_pool_padding,
                                                   self.cnn_para.l01_pool_stride,
                                                   self.cnn_para.l01_pool_ksize,

                                                   self.cnn_para.l01_is_drop,
                                                   self.cnn_para.l01_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )   
        ### ======== LAYER 02
        with tf.device('/gpu:0'), tf.variable_scope("conv02")as scope:
             [self.output_layer02, self.mid_layer02] = self.conv_layer(
                                                   self.output_layer01,

                                                   self.cnn_para.l02_filter_height,
                                                   self.cnn_para.l02_filter_width,
                                                   self.cnn_para.l02_pre_filter_num,
                                                   self.cnn_para.l02_filter_num,
                                                   self.cnn_para.l02_conv_padding,
                                                   self.cnn_para.l02_conv_stride,

                                                   self.cnn_para.l02_is_norm,

                                                   self.cnn_para.l02_conv_act_func,

                                                   self.cnn_para.l02_is_pool,
                                                   self.cnn_para.l02_pool_type,
                                                   self.cnn_para.l02_pool_padding,
                                                   self.cnn_para.l02_pool_stride,
                                                   self.cnn_para.l02_pool_ksize,

                                                   self.cnn_para.l02_is_drop,
                                                   self.cnn_para.l02_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )   

        ### ======== LAYER 03
        with tf.device('/gpu:0'), tf.variable_scope("conv03")as scope:
             [self.output_layer03, self.mid_layer03] = self.conv_layer(
                                                   self.output_layer02,

                                                   self.cnn_para.l03_filter_height,
                                                   self.cnn_para.l03_filter_width,
                                                   self.cnn_para.l03_pre_filter_num,
                                                   self.cnn_para.l03_filter_num,
                                                   self.cnn_para.l03_conv_padding,
                                                   self.cnn_para.l03_conv_stride,

                                                   self.cnn_para.l03_is_norm,

                                                   self.cnn_para.l03_conv_act_func,

                                                   self.cnn_para.l03_is_pool,
                                                   self.cnn_para.l03_pool_type,
                                                   self.cnn_para.l03_pool_padding,
                                                   self.cnn_para.l03_pool_stride,
                                                   self.cnn_para.l03_pool_ksize,

                                                   self.cnn_para.l03_is_drop,
                                                   self.cnn_para.l03_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )   

        ### ======== LAYER 04
        with tf.device('/gpu:0'), tf.variable_scope("conv04")as scope:
             [self.output_layer04, self.mid_layer04] = self.conv_layer(
                                                   self.output_layer03,

                                                   self.cnn_para.l04_filter_height,
                                                   self.cnn_para.l04_filter_width,
                                                   self.cnn_para.l04_pre_filter_num,
                                                   self.cnn_para.l04_filter_num,
                                                   self.cnn_para.l04_conv_padding,
                                                   self.cnn_para.l04_conv_stride,

                                                   self.cnn_para.l04_is_norm,

                                                   self.cnn_para.l04_conv_act_func,

                                                   self.cnn_para.l04_is_pool,
                                                   self.cnn_para.l04_pool_type,
                                                   self.cnn_para.l04_pool_padding,
                                                   self.cnn_para.l04_pool_stride,
                                                   self.cnn_para.l04_pool_ksize,

                                                   self.cnn_para.l04_is_drop,
                                                   self.cnn_para.l04_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  ) 

             self.final_output = self.output_layer04

###==================================================== OTHER FUNCTION ============================
    #02/ CONV LAYER
    def conv_layer(self, 
                  input_value, 

                  filter_height, 
                  filter_width, 
                  pre_filter_num, 
                  filter_num, 
                  conv_padding, 
                  conv_stride,

                  is_norm,

                  act_func,

                  is_pool, 
                  pool_type, 
                  pool_padding, 
                  pool_stride, 
                  pool_ksize, 

                  is_drop,
                  drop_prob,

                  mode,
                  scope=None
                 ):
        #------------------------------#
        def reduce_var(x, axis=None, keepdims=False, name=None):
            m = tf.reduce_mean(x, axis=axis, keepdims=True, name=name) #keep same dimension for subtraction
            devs_squared = tf.square(x - m)
            return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims, name=name)
        
        def reduce_std(x, axis=None, keepdims=False, name=None):
            return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims, name=name))

        #------------------------------#

        with tf.variable_scope(scope or 'conv-layer') as scope:

            # shape: [5,5,1,32] or [5,5,32,64]
            filter_shape = [filter_height, filter_width, pre_filter_num, filter_num]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")   # this is kernel 
            b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")

            #Convolution layer 
            conv_output = tf.nn.conv2d(
                                 input_value,
                                 W,
                                 strides = conv_stride,
                                 padding = conv_padding,
                                 name="conv"
                                 )  #default: data format = NHWC


            #Active function layer
            if (act_func == 'RELU'):
                act_func_output = tf.nn.relu(tf.nn.bias_add(conv_output, b), name="RELU")
            elif (act_func == 'TANH'):
                act_func_output = tf.nn.tanh(tf.nn.bias_add(conv_output, b), name="TANH")

            #BachNorm Layer
            if(is_norm == True):
                batch_output = tf.contrib.layers.batch_norm(
                                                             act_func_output, 
                                                             is_training = mode, 
                                                             decay = 0.9,
                                                             zero_debias_moving_mean=True
                                                           )
            else:     
                batch_output = act_func_output

            #Pooling layer
            if(is_pool == True):
                if (pool_type == 'MEAN'):
                    pool_output = tf.nn.avg_pool(
                                          batch_output,
                                          ksize   = pool_ksize,   
                                          strides = pool_stride,
                                          padding = pool_padding,
                                          name="mean_pool"
                                         )
                elif (pool_type == 'MAX'):
                    pool_output = tf.nn.max_pool(
                                          batch_output,
                                          ksize   = pool_ksize,   
                                          strides = pool_stride,
                                          padding = pool_padding,
                                          name="max_pool"
                                         )
                elif (pool_type == 'GLOBAL_MAX'):
                    pool_output = tf.reduce_max(
                                          batch_output,
                                          axis=[1,2],
                                          name='global_max'
                                         )
                elif (pool_type == 'GLOBAL_MEAN'):
                    pool_output = tf.reduce_mean(
                                          batch_output,
                                          axis=[1,2],
                                          name='global_moment01_pool'
                                         )
                elif (pool_type == 'GLOBAL_STD'):   #only for testing (not apply for training)
                    pool_output = reduce_std(
                                          batch_output,
                                          axis=[1,2],
                                          name = "global_moment02_pool"
                                         )
                    #print pool_output.get_shape()
                    #exit()
            else:
                pool_output = batch_output

            #Dropout
            if(is_drop == True):
                drop_output = tf.layers.dropout(
                                                pool_output, 
                                                rate = drop_prob,
                                                training = mode,
                                                name = 'Dropout'
                                               )
            else:     
                drop_output = pool_output

            
            mid_output = tf.reduce_mean(batch_output,
                                       axis=[1,2],
                                       name='global_mean'
                                      )

            return drop_output, mid_output

