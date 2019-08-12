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
from sklearn import datasets, svm, metrics
from numpy import random

from model_conf import *

#import data_helpers
#from shutil import copyfile
#import h5py


#=========================================== 01/ PARAMETERS
#is_training = sys.argv[1]
#is_validating = sys.argv[2]
random.seed(1)
print("\n ==================================================================== SETUP PARAMETERS...")

# 1.1/ Directory 
tf.flags.DEFINE_string("TRAIN_DIR",  "./../11_01_run/data/post_train_data/", "Directory of feature")
tf.flags.DEFINE_string("VALID_DIR",  "./../11_01_run/data/mid_test_extr/", "Directory of feature")


tf.flags.DEFINE_string("OUT_DIR",    "./data/",     "Point to output directory")

# 1.2/ Training para 
tf.flags.DEFINE_integer("N_TRAIN_MUL_BATCH", 1,     "Multi Batch Number for Training")   #74
tf.flags.DEFINE_integer("BATCH_SIZE",        100,    "Batch Size ")
tf.flags.DEFINE_integer("NUM_EPOCHS",        500,    "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("N_CLASS",           15,     "Class Number")
tf.flags.DEFINE_integer("N_VALID",           390,   "Valid file number")

tf.flags.DEFINE_integer("TESTING_EVERY",     200,   "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("CHECKPOINT_EVERY",  100,   "Save model after this many steps (default: 100)")
tf.flags.DEFINE_float  ("LEARNING_RATE",     1e-4,  "Learning rate")

# 1.3/ Device Report Para
tf.flags.DEFINE_boolean("allow_soft_placement", True,  "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
overfitting_cnt = 0
mixup_num = FLAGS.BATCH_SIZE/2
#======================================================  02/ HANDLE FILE
### train dir
train_dir = os.path.abspath(FLAGS.TRAIN_DIR)
org_train_file_list = os.listdir(train_dir)
train_file_list = []  #remove .file
for nFileTrain in range(0,len(org_train_file_list)):
    isHidden=re.match("\.",org_train_file_list[nFileTrain])
    if (isHidden is None):
        train_file_list.append(org_train_file_list[nFileTrain])
train_file_list = sorted(train_file_list)        

### valid dir
valid_dir = os.path.abspath(FLAGS.VALID_DIR)
org_valid_file_list = os.listdir(valid_dir)
valid_file_list = []  #remove .file
for nClassValid in range(0,len(org_valid_file_list)):
    isHidden=re.match("\.",org_valid_file_list[nClassValid])
    if (isHidden is None):
        valid_file_list.append(org_valid_file_list[nClassValid])
valid_file_num  = len(valid_file_list)
valid_file_list = sorted(valid_file_list)


#======================================================  03/ TRAINING & SAVE
print("\n ==================================================================== TRAINING DATA...")

tf.reset_default_graph()
with tf.Graph().as_default():
    tf.set_random_seed(1)
    session_conf = tf.ConfigProto( allow_soft_placement=FLAGS.allow_soft_placement, 
                                   log_device_placement=FLAGS.log_device_placement
                                 )

    sess = tf.Session(config=session_conf)

    with sess.as_default():

        # ==================================================  01/ instance network model
        print("\n =============== 01/ Instance Model")
        model = model_conf()  

        # 02/ Define Training procedure, optional optimizer ....
        print("\n =============== 02/ Setting Training Options")
        print("\n + Adam optimizer ")
        print("\n + Learning Rate: {}".format(FLAGS.LEARNING_RATE))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step    = tf.Variable(0, name="global_step", trainable=False)
            optimizer      = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op       = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # ====================================================  02/ Setup kinds of report summary 
        print("\n =============== 03/ Setting Report ...")
        print("\n + Gradient ")
        print("\n + Sparsity ")
        print("\n + Loss ")
        print("\n + Accuracy ")

        # ====================================================   03/ Setup training summary directory
        print("\n =============== 04/ Setting Directory for Saving...")
        stored_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.OUT_DIR))
        print("+ Writing to {}\n".format(stored_dir))

        train_summary_dir = os.path.join(stored_dir, "summaries", "train")   #stored_dir/summaries/train
        print("+ Training summary Writing to {}\n".format(train_summary_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(stored_dir, "checkpoints"))
        print("XXXXXXXXXXXXXXXXX: Checkpoint Dir: {}\n".format(checkpoint_dir))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        best_model_dir = os.path.join(stored_dir, "model")
        print("XXXXXXXXXXXXXXXXX: Best model Dir: {}\n".format(best_model_dir))
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)


        ### ======================================================= 04/ Save and initial/load best model
        # Create saver     
        print("\n =============== 05/ Creating Saver...")
        saver = tf.train.Saver(tf.global_variables())

        # Load saved model to continue training or initialize all variables for new Model
        best_model_files     = os.path.join(best_model_dir, "best_model")
        best_model_meta_file = os.path.join(best_model_dir, "best_model.meta")
        print("XXXXXXXXXXXXXXXXX: Best Model Files: {}\n".format(best_model_files))
        print("XXXXXXXXXXXXXXXXX: Best Model Meta File: {}\n".format(best_model_meta_file))

        if os.path.isfile(best_model_meta_file):
            print("\n=============== 06/ Latest Model Loaded from dir: {}" .format(best_model_dir))
            saver = tf.train.import_meta_graph(best_model_meta_file)
            saver.restore(sess, best_model_files)
        else:
            print("\n=============== 06/ New Model Initialized")
            sess.run(tf.global_variables_initializer())

        # ============================================================ 05/ Define training function that is called every epoch
        def train_process(x_train_batch, y_train_batch):
            # Training every batch

            feed_dict= {model.input_layer_val:   x_train_batch,
                        model.expected_classes:  y_train_batch,
                        model.mode: True
                       }

            # Remove summary report
            [ _, step, loss, accuracy, end_output] = sess.run([train_op, global_step, model.loss, model.accuracy, model.output_layer], feed_dict)

            time_str = datetime.datetime.now().isoformat()
            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))    #pint here --> no need returna

            return accuracy, end_output

        def test_process(x_test_batch):
            # Training every batch

            feed_dict= {model.input_layer_val:   x_test_batch,
                        model.mode: False
                       }

            # Training and return data
            [step, end_output] = sess.run([global_step, model.prob_output_layer], feed_dict)

            return end_output

        def get_train_batch(nTrainMulBatch):
            #print ("========== Obtain train data from the multibatch : {:g}".format(nTrainMulBatch + 1))

            train_file = train_dir + '/' + train_file_list[nTrainMulBatch]
            data_train        = np.load(train_file)     
            x_train_mul_batch = data_train['seq_x']
            y_train_mul_batch = data_train['seq_y']

            [nS, nC] = x_train_mul_batch.shape
            train_mul_batch_num = nS

            return x_train_mul_batch, y_train_mul_batch, train_mul_batch_num
        
        def get_test_batch(test_file_dir):

            data_test    = np.load(test_file_dir)     
            x_test_batch = data_test
  
            return x_test_batch

        ### ============================================================  06/ Call epoch, train, validate, and test
        is_training   = 1  
        is_validating = 1

        start_multi_batch = 0
        is_break = 0

        if(is_training):
            ### Every Epoch
            for nEpoch in range(FLAGS.NUM_EPOCHS):
                if (is_break == 1):
                    break
                print("\n=======================  Epoch is", nEpoch, ";============================")
                #Every multiple batch
                for nTrainMulBatch in range(start_multi_batch, int(FLAGS.N_TRAIN_MUL_BATCH)):
                    if (is_break == 1):
                        break
                    # get data for every batch
                    [x_train_mul_batch, y_train_mul_batch, train_mul_batch_num] = get_train_batch(nTrainMulBatch)

                    #Every batch  in multi-batches
                    for nBatch in range(int(train_mul_batch_num/FLAGS.BATCH_SIZE)):
                        if (is_break == 1):
                            break

                        stPt = nBatch*FLAGS.BATCH_SIZE
                        edPt   = (nBatch+1)*FLAGS.BATCH_SIZE 

                        x_train_batch = x_train_mul_batch[stPt:edPt, :]
                        y_train_batch = y_train_mul_batch[stPt:edPt, :]

                        # Mixture here 
                        X1      = x_train_batch[0:50,:]
                        X2      = x_train_batch[50:100,:]
                        y1      = y_train_batch[0:50,:]
                        y2      = y_train_batch[50:100,:]

                        # Betal dis
                        b   = np.random.beta(0.4, 0.4, 50)
                        X_b = b.reshape(50, 1)
                        y_b = b.reshape(50, 1)

                        xb_mix   = X1*X_b     + X2*(1-X_b)
                        xb_mix_2 = X1*(1-X_b) + X2*X_b
                        yb_mix   = y1*y_b     + y2*(1-y_b)
                        yb_mix_2 = y1*(1-y_b) + y2*y_b
     
                        # Uniform dis
                        l   = np.random.random(50)
                        X_l = l.reshape(50, 1)
                        y_l = l.reshape(50, 1)

                        xl_mix   = X1*X_l     + X2*(1-X_l)
                        xl_mix_2 = X1*(1-X_l) + X2*X_l
                        yl_mix   = y1* y_l    + y2 * (1-y_l)
                        yl_mix_2 = y1*(1-y_l) + y2*y_l

                        x_train_batch = np.concatenate((xb_mix, X1, xl_mix, xb_mix_2, X2, xl_mix_2), 0)   
                        y_train_batch = np.concatenate((yb_mix, y1, yl_mix, yb_mix_2, y2, yl_mix_2), 0)
                        
                        # Call training process
                        train_acc, train_end_output = train_process(x_train_batch, y_train_batch)

                        # At check poit, verify the accuracy of testing set, and update the best model due to accuracy
                        current_step = tf.train.global_step(sess, global_step)
                        if (current_step % FLAGS.CHECKPOINT_EVERY == 0):  
                            print("Total Data Training Accuracy At Step {}: {}".format(current_step, train_acc))
                            with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                                text_file.write("{0}\n".format(train_acc))

                            if(nEpoch == FLAGS.NUM_EPOCHS-3):         
                                # Store the best model    
                                best_model_files = os.path.join(best_model_dir, "best_model")
                                saved_path       = saver.save(sess, best_model_files)
                                print("Saved best model during testing to {} at batch {}\n".format(saved_path, current_step))
                                print("Break at Final Epoch: {}" .format(current_step))
                                is_break = 1
                    #for nBatch in multi-batches
                #for multi-batches
            #for epoch    
                
        if (is_validating == 1):  #if is_training==0  --> testing
            file_valid_acc   = 0
            fuse_matrix      = np.zeros([FLAGS.N_CLASS, FLAGS.N_CLASS])
            valid_metric_reg = np.zeros(FLAGS.N_VALID)
            valid_metric_exp = np.zeros(FLAGS.N_VALID)

            nTotalValidFile = 0

            for nFileValid in range(0,valid_file_num):

                valid_file_name = valid_file_list[nFileValid]
                valid_file_dir  = valid_dir + '/' + valid_file_name
                x_valid_batch   = get_test_batch(valid_file_dir)
            
                # Call training process
                valid_end_output = test_process(x_valid_batch)
            
                # Compute acc
                sum_valid_end_output = np.sum(valid_end_output, axis=0) #1xnClass
                valid_res_reg        = np.argmax(sum_valid_end_output)

                if(re.match("beach_", valid_file_name)):
                    valid_res_exp  = 0
                elif(re.match("bus_", valid_file_name)):
                    valid_res_exp  = 1
                elif(re.match("cafe_restaurant_", valid_file_name)):
                    valid_res_exp  = 2
                elif(re.match("car_", valid_file_name)):
                    valid_res_exp  = 3
                elif(re.match("city_center_", valid_file_name)):
                    valid_res_exp  = 4
                elif(re.match("forest_path_", valid_file_name)):
                    valid_res_exp  = 5
                elif(re.match("grocery_store_", valid_file_name)):
                    valid_res_exp  = 6
                elif(re.match("home_", valid_file_name)):
                    valid_res_exp  = 7
                elif(re.match("library_", valid_file_name)):
                    valid_res_exp  = 8
                elif(re.match("metro_station_", valid_file_name)):
                    valid_res_exp  = 9
                elif(re.match("office_", valid_file_name)):
                    valid_res_exp  = 10
                elif(re.match("park_", valid_file_name)):
                    valid_res_exp  = 11
                elif(re.match("residential_area_", valid_file_name)):
                    valid_res_exp  = 12
                elif(re.match("train_", valid_file_name)):
                    valid_res_exp  = 13
                elif(re.match("tram_", valid_file_name)):
                    valid_res_exp  = 14

                valid_metric_reg[nTotalValidFile] = int(valid_res_reg)
                valid_metric_exp[nTotalValidFile] = int(valid_res_exp)
                nTotalValidFile = nTotalValidFile + 1

                #For general report
                fuse_matrix[valid_res_exp, valid_res_reg] = fuse_matrix[valid_res_exp, valid_res_reg] + 1
                if(valid_res_reg == valid_res_exp):
                    file_valid_acc = file_valid_acc + 1 
               
            # For general report
            file_valid_acc  = file_valid_acc*100/nTotalValidFile
            print("Testing Accuracy: {} % \n".format(file_valid_acc))   

            #for sklearn metric
            print("Classification report for classifier \n%s\n"
                  % (metrics.classification_report(valid_metric_exp, valid_metric_reg)))
            cm = metrics.confusion_matrix(valid_metric_exp, valid_metric_reg)
            print("Confusion matrix:\n%s" % cm)

            with open(os.path.join(stored_dir,"valid_acc_log.txt"), "a") as text_file:
                text_file.write("========================== VALIDATING ONLY =========================================== \n\n")
                text_file.write("On File Final Accuracy:  {}%\n".format(file_valid_acc))
                text_file.write("{0} \n".format(fuse_matrix))
                text_file.write("========================================================================== \n\n")
