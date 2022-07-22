from sympy import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import pandas as pd
from rich import print
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,recall_score,confusion_matrix


class ClassificationLog(object):
    def __init__(self,name):
        self.name = name
        self.history = {
            "recall"  : [],
            "f1-score": [],
            "accuracy": [],
            "loss"    : []   
        }
        self.sum_recall = 0.0
        self.sum_acc  = 0.0
        self.sum_loss = 0.0
        self.sum_f1 = 0.0
        self.n  = 0
        self.confusion_matrix = None
    def reset(self):
        self.sum_acc = 0.0
        self.sum_recall = 0.0
        self.sum_loss = 0.0
        self.sum_f1 = 0.0
        self.n = 0

    def Average(self):
        return [
            self.sum_recall/self.n,
            self.sum_f1/self.n,
            self.sum_acc/self.n,
            self.sum_loss/self.n
        ]
    def update_confusion_matrix(self,m):
        self.confusion_matrix = np.copy(m)
    def update(self,recall,f1,accuracy,loss):
        self.sum_recall += recall
        self.sum_f1 += f1
        self.sum_acc += accuracy
        self.sum_loss += loss
        self.n += 1
        self.history['recall'].append(recall)
        self.history['accuracy'].append(accuracy)
        self.history['loss'].append(loss)
        self.history['f1-score'].append(f1)
    
    def to_csv(self,name):
        pd.DataFrame(self.history).to_csv(name,index = None)
        if not(self.confusion_matrix is None ):
            np.save(f"{name.split('.')[0]}.npy",self.confusion_matrix)
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()



def compute_metrics(true,predict):
    labels = [i for i in range(config.parameter['num_classes'])]
    return(recall_score(true,predict,average = "weighted",labels=labels,zero_division=0),
           f1_score(true,predict,average = "weighted",labels=labels,zero_division=0),
           accuracy_score(true,predict))



def compute_confusion_matrix(model,loader):
    model.eval()
    predicts = []
    true = []
    for img,label in loader:
        out = model(img)
        predict = out.argmax(dim = -1)
        predict = to_numpy(predict).tolist()
        predicts += predict
        true += to_numpy(label).tolist()
    return confusion_matrix(true,predict)

