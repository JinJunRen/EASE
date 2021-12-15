# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:05:39 2020
           hi=+1     hi=-1
hj=+1      a(=TP)    c(=FN)
hi=-1      b(=FP)    d(=TN)
@author: Administrator
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class DiversityMeasure:
    def __init__(self,true,pred):
        conf_m=confusion_matrix(true,pred)
        self.lags = False
        # print(f"conf_m's size:{conf_m.shape}")
        if conf_m.shape == (1,1):
            self.lags = True
            return None
        self.d=conf_m[0,0]
        self.a=conf_m[1,1]
        self.b=conf_m[0,1]
        self.c=conf_m[1,0]
        self.m=conf_m.sum()
        
    def disagreement(self):
        return (self.b+self.c)/(self.m*1.0)
    
    def correlation(self):
        upelement=self.a*self.d-self.b*self.c
        denominator=np.sqrt((self.a+self.b)*(self.a+self.c)*(self.c+self.d)*(self.b+self.d))
        return upelement/denominator
    
    def q_statistic(self):
        upelement=self.a*self.d-self.b*self.c
        denominator=self.a*self.d+self.b*self.c
        return (upelement*1.0)/denominator
    
    def k_statistic(self):
        p1=(self.a+self.d)/(self.m*1.0)
        p2=((self.a+self.b)*(self.a+self.c)+(self.c+self.d)*(self.b+self.d))/(self.m**2*1.0)
        return (p1-p2)/(1-p2)
        

if __name__ == '__main__':
    y1=[1,1,0,0,1,0,1]
    y2=[1,0,1,0,0,1,1]
    d=DiversityMeasure(y1,y2)
    print(f"disagreement:{d.disagreement()},correlation:{d.correlation()},Q:{d.q_statistic()},k:{d.k_statistic()}")
    
    
    
        
        
    