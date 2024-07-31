from __future__ import print_function

import numpy as np
import torch
from types import FunctionType
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter  
import torch.nn as nn
from scipy.spatial.distance import cdist

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
 
    def __call__(self, img):
        img = np.array(img)
        c, h, w = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(c, h, w))
        img = N + img
        img = torch.tensor(img)
        return img

def discrepancy(self, out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[],save_filename=None):
    plt.figure(figsize=(10, 6),dpi=600)
    plt.imshow(cm*100, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90,size=12)
    plt.yticks(tick_marks, labels,size=12)
    for i in range(len(tick_marks)):
        for j in range(len(tick_marks)):
            if i!=j:
                text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=10)
            elif i==j:
                if int(np.around(cm[i,j]*100))==100:
                    text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=7,color='darkorange')
                else:
                    text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=10,color='darkorange')
            

    plt.tight_layout()
    # plt.ylabel('True label',fontdict={'size':8,})
    # plt.xlabel('Predicted label',fontdict={'size':8,})
    if save_filename is not None:
        plt.savefig(save_filename,dpi=1200,bbox_inches = 'tight')
    plt.close()