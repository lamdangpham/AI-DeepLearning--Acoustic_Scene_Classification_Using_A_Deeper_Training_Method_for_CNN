import numpy as np
import os
import argparse
import math
import scipy.io
import re
import time
import datetime
from scipy import stats


#======================================================  01/ DIRECTORY 
out_dir    = "./data/"
source_dir = "./data/mid_train_extr/" #
class_num  = 15; #

#For storing data for train 02
stored_dir = os.path.abspath(os.path.join(os.path.curdir,out_dir))
if not os.path.exists(stored_dir):
    os.makedirs(stored_dir)
des_dir = os.path.abspath(os.path.join(stored_dir, "post_train_data")) #
if not os.path.exists(des_dir):
    os.makedirs(des_dir)


#======================================================  03/ HANDLE SOURCE FILE
source_dir = os.path.abspath(source_dir) 
org_source_file_list = os.listdir(source_dir)
source_file_list = []  #remove .file
for nFileSc in range(0,len(org_source_file_list)):
    isHidden=re.match("\.",org_source_file_list[nFileSc])
    if (isHidden is None):
        source_file_list.append(org_source_file_list[nFileSc])
source_file_num  = len(source_file_list)
source_file_list = sorted(source_file_list)


# Destination file
#draw_des_file = os.path.abspath(os.path.join(stored_dir, "train_draw_file"))
des_file = os.path.abspath(os.path.join(des_dir, "mul_batch_train"))

#======================================================  04/ COLLECT DATA into 22 GROUP
for nFile in range(int(source_file_num)):
#for nFile in range(0,4):

    source_file_name = source_file_list[nFile]
    if   "Class_0_" in source_file_name:
        nClass = 0
    elif "Class_1_" in source_file_name:     
        nClass = 1
    elif "Class_2_" in source_file_name:     
        nClass = 2
    elif "Class_3_" in source_file_name:     
        nClass = 3
    elif "Class_4_" in source_file_name:     
        nClass = 4
    elif "Class_5_" in source_file_name:     
        nClass = 5
    elif "Class_6_" in source_file_name:     
        nClass = 6
    elif "Class_7_" in source_file_name:     
        nClass = 7
    elif "Class_8_" in source_file_name:     
        nClass = 8
    elif "Class_9_" in source_file_name:     
        nClass = 9
    elif "Class_10_" in source_file_name:     
        nClass = 10
    elif "Class_11_" in source_file_name:     
        nClass = 11
    elif "Class_12_" in source_file_name:     
        nClass = 12
    elif "Class_13_" in source_file_name:     
        nClass = 13
    elif "Class_14_" in source_file_name:     
        nClass = 14


    # Open file
    file_open = source_dir + '/' + source_file_name
    file_str  = np.load(file_open)   #31x256
    #print file_str
    #exit()

    [feat_num, col_num] = np.shape(file_str)

    seq_y = np.zeros((feat_num,class_num))
    for i in range(feat_num):
        seq_y[i,nClass] = 1    

    # For drawing
    if (nFile == 0):
        mul_seq_x  =  file_str
        mul_seq_y  =  seq_y
    else:    
        mul_seq_x  =  np.concatenate((mul_seq_x, file_str), axis=0)
        mul_seq_y  =  np.concatenate((mul_seq_y, seq_y),    axis=0)

# multiple of 100
[row_num, col_num] = np.shape(mul_seq_x)
dim = int(np.ceil(row_num/100 + 1)*100)
for i in range(dim - row_num):
    rd_ps     = np.random.randint(1,row_num)
    ins_seq_x = mul_seq_x[rd_ps,:]
    ins_seq_y = mul_seq_y[rd_ps,:]

    ins_seq_x = np.reshape(ins_seq_x,[1,col_num])
    ins_seq_y = np.reshape(ins_seq_y,[1,class_num])

    mul_seq_x = np.concatenate((mul_seq_x, ins_seq_x), axis=0)
    mul_seq_y = np.concatenate((mul_seq_y, ins_seq_y), axis=0)

# Random positon
new_mul_seq_x = np.zeros((dim, col_num))
new_mul_seq_y = np.zeros((dim, class_num))

kk = np.random.permutation(dim)
for i in range(dim):
    new_mul_seq_x[i,:]   = mul_seq_x[kk[i],:]
    new_mul_seq_y[i,:]   = mul_seq_y[kk[i],:]
# Save file
np.savez(des_file, seq_x=new_mul_seq_x, seq_y=new_mul_seq_y)

