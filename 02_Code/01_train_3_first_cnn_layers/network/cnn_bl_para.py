
#import tensorflow as tf
import numpy as np
import os

class cnn_bl_para(object):

    def __init__(self):

        #====================================Layer 01: conv 
        #conv
        self.l01_filter_height  = 9
        self.l01_filter_width   = 9
        self.l01_pre_filter_num = 1
        self.l01_filter_num     = 32
        self.l01_conv_padding   = 'SAME'        #SAME: zero padding; VALID: without padding  
        self.l01_conv_stride    = [1,1,1,1]      
        #batch
        self.l01_is_norm        = True
        #act
        self.l01_conv_act_func  = 'RELU'
        #pool
        self.l01_is_pool        = True   
        self.l01_pool_type      = 'MAX'
        self.l01_pool_padding   = 'VALID' 
        self.l01_pool_stride    = [1,2,2,1]
        self.l01_pool_ksize     = [1,2,2,1]
        #drop
        self.l01_is_drop        = True
        self.l01_drop_prob      = 0.1

        #======================================Layer 02: conv 
        #conv
        self.l02_filter_height  = 7
        self.l02_filter_width   = 7
        self.l02_pre_filter_num = 32
        self.l02_filter_num     = 64
        self.l02_conv_padding   = 'SAME' 
        self.l02_conv_stride    = [1,1,1,1]      
        #batch
        self.l02_is_norm        = True
        #act
        self.l02_conv_act_func  = 'RELU'
        #pool
        self.l02_is_pool        = True  
        self.l02_pool_type      = 'MAX' 
        self.l02_pool_padding   = 'VALID' 
        self.l02_pool_stride    = [1,2,2,1]
        self.l02_pool_ksize     = [1,2,2,1]
        #drop
        self.l02_is_drop        = True
        self.l02_drop_prob      = 0.1

        #======================================Layer 03: conv 
        #conv
        self.l03_filter_height  = 5
        self.l03_filter_width   = 5
        self.l03_pre_filter_num = 64
        self.l03_filter_num     = 128
        self.l03_conv_padding   = 'SAME' 
        self.l03_conv_stride    = [1,1,1,1]      
        #batch
        self.l03_is_norm        = True
        #act
        self.l03_conv_act_func  = 'RELU'
        #pool
        self.l03_is_pool        = True  
        self.l03_pool_type      = 'MAX'
        self.l03_pool_padding   = 'VALID' 
        self.l03_pool_stride    = [1,2,2,1]
        self.l03_pool_ksize     = [1,2,2,1]
        #drop
        self.l03_is_drop        = True
        self.l03_drop_prob      = 0.2
        
        #======================================Layer 04: conv 
        #conv
        self.l04_filter_height  = 3
        self.l04_filter_width   = 3
        self.l04_pre_filter_num = 128
        self.l04_filter_num     = 256
        self.l04_conv_padding   = 'SAME' 
        self.l04_conv_stride    = [1,1,1,1]      
        #batch
        self.l04_is_norm        = True
        #act
        self.l04_conv_act_func  = 'RELU'
        #pool
        self.l04_is_pool        = True 
        self.l04_pool_type      = 'GLOBAL_MEAN' 
        self.l04_pool_padding   = 'VALID' 
        self.l04_pool_stride    = [1,2,2,1]
        self.l04_pool_ksize     = [1,2,2,1]
        #drop
        self.l04_is_drop        = True
        self.l04_drop_prob      = 0.2
