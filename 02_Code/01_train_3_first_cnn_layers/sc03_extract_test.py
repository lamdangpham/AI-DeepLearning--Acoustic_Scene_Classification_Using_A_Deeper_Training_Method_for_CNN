import tensorflow as tf
import numpy as np
import os
import argparse
import math
import scipy.io
#from scipy.io import loadmat
import re
import time
import datetime
import sys
#import data_helpers
#from shutil import copyfile
#import h5py

sys.path.append('./network/')
from model_conf import *

#===============AAaAaa=aas:2 ============================ 01/ PARAMETERS
print("\n ==================================================================== SETUP PARAMETERS...")

# 1.1/ Directory 
tf.flags.DEFINE_string("TEST_DIR",      "./../input_data/data_test/",       "Directory of feature")
tf.flags.DEFINE_string("OUT_DIR",       "./data/",                                   "Point to output directory")

tf.flags.DEFINE_integer("N_CLASS",           15,     "Class Number")
tf.flags.DEFINE_integer("N_VALID",           390,   "Valid file number")

# 1.3/ Device Report Para
tf.flags.DEFINE_boolean("allow_soft_placement", True,  "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


# 1.4/ Report & Check Para
FLAGS = tf.flags.FLAGS

#======================================================  02/ HANDLE FILE
test_dir = os.path.abspath(FLAGS.TEST_DIR)
org_test_file_list = os.listdir(test_dir)
test_file_list = []  #remove .file
for nClassTest in range(0,len(org_test_file_list)):
    isHidden=re.match("\.",org_test_file_list[nClassTest])
    if (isHidden is None):
        test_file_list.append(org_test_file_list[nClassTest])
test_file_num  = len(test_file_list)
test_file_list = sorted(test_file_list)

#======================================================  03/ TESTING

with tf.Graph().as_default():
    session_conf = tf.ConfigProto( allow_soft_placement=FLAGS.allow_soft_placement, 
                                   log_device_placement=FLAGS.log_device_placement
                                 )
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        model = model_conf()  
        global_step    = tf.Variable(0, name="global_step", trainable=False)

        # ====================================================   03/ Setup training summary directory
        print("\n =============== 04/ Setting Directory for Saving...")
        stored_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.OUT_DIR))
        best_model_dir = os.path.join(stored_dir, "model")   #
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        #========================================================  04/ FOR STORING RESULT DATA 
        mid_layer_test_dir = os.path.abspath(os.path.join(stored_dir, "mid_test_extr"))
        if not os.path.exists(mid_layer_test_dir):
            os.makedirs(mid_layer_test_dir)

        ### ======================================================= 05/ Save and initial/load best model
        # Create saver     
        print("\n =============== 05/ Creating Saver...")
        saver = tf.train.Saver(tf.global_variables())

        # Load saved model to continue training or initialize all variables for new Model
        best_model_files     = os.path.join(best_model_dir, "best_model")
        best_model_meta_file = os.path.join(best_model_dir, "best_model.meta")
        if os.path.isfile(best_model_meta_file):
            print("\n=============== 06/ Latest Model Loaded from dir: {}" .format(best_model_dir))
            saver = tf.train.import_meta_graph(best_model_meta_file)
            saver.restore(sess, best_model_files)
        else:
            print("\n=============== 06/ New Model Initialized")
            sess.run(tf.global_variables_initializer())

        # ============================================================ 06/ Define training function that is called every epoch
        def test_process(x_test_batch):
            feed_dict= {model.input_layer_val:   x_test_batch,
                        model.mode: False
                       }
            [step, wanted_data] = sess.run([global_step, model.wanted_data], feed_dict)

            return wanted_data

        def get_test_batch(test_file_dir):
            data_test    = np.load(test_file_dir)     
            x_test_batch = data_test['seq_x']
            [nS, nF, nT] = x_test_batch.shape
            x_test_batch = np.reshape(x_test_batch, [nS,nF,nT,1])      

            return x_test_batch

        ### ============================================================  07/ Call epoch, train and test
        for nFileTest in range(0,test_file_num):
        #for nFileTest in range(0,5):
            file_name_open = test_file_list[nFileTest]
            test_file_dir = test_dir + '/' + file_name_open

            # Get batch
            x_test_batch = get_test_batch(test_file_dir)

            # Call testing process
            wanted_data = test_process(x_test_batch)

            # Store result into matrix
            new_file_name = os.path.splitext(file_name_open)[0]
            save_file_dir = mid_layer_test_dir + '/' + new_file_name
            np.save(save_file_dir, wanted_data)
