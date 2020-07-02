# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:28:39 2020

@author: Mouiad
"""

from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import precision_recall_fscore_support
y_true=r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\data\CULane\driver_100_30frame\05250419_0290.MP4\00030.lines.txt"
y_pred=r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\prob2lines\output\vgg_SCNN_DULR_w9\driver_100_30frame\05251517_0433.MP4\00030.lines.txt"

# Read the data




with open(y_true, 'r') as infile:
    actual = [str(i) for i in infile]
s=""
for i in actual:
    s+=i
y_real=s.split()
XX=[]
for i in y_real:
   XX.append(float(i)) 
   
X=[]
for i in XX:
   X.append(int(i))

with open(y_pred, 'r') as infile:
    predicted = [str(i) for i in infile]

ss=""
for i in predicted:
    ss+=i
y_pred=ss.split()
YY=[]
for i in y_pred:
   YY.append(float(i))

# Make confusion matrix
X=np.array(X)
YY=np.array(YY)
Y=np.resize(YY,(160,1))
Y=Y>500
confusion = confusion_matrix(X, Y)


