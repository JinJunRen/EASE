# -*- coding: utf-8 -*-
"""
@author: JinJunRen
"""
import sys
import os
import argparse
import time
sys.path.append("..")
import pandas as pd
import tools.dataprocess as dp
from ensemble.self_paced_ensemble import SelfPacedEnsemble
from ensemble.equalizationensemble import EASE
from ensemble.ECUBoost_RF import ECUBoostRF
from ensemble.hub_ensemble import HashBasedUndersamplingEnsemble
from ensemble.canonical_ensemble import *
import numpy as np
from tools.imbalancedmetrics import ImBinaryMetric
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier 

METHODS = ['EASE','SMOTEBoost', 'SMOTEBoost' , 'SMOTEBagging', 'RUSBoost', 'UnderBagging', 'BalanceCascade']
RANDOM_STATE = None


def parse():
    '''Parse system arguments.'''
    parse=argparse.ArgumentParser(
        description='General excuting Ensemble method', 
        usage='genealrunEnsemble.py -dir <datasetpath> -alg <algorithm name> -est <number of estimators>'
        ' -n <n-fold>'
        )
    parse.add_argument("-dir",dest="dataset",help="path of the datasets or a dataset")
    parse.add_argument("-alg",dest="algorithm",nargs='+',help="list of the algorithm names ")
    parse.add_argument("-est",dest="estimators",default=10, type=int, help="number of estimators")
    parse.add_argument("-n",dest="nfold",default=5, type=int, help="n fold")
    return parse.parse_args()


def init_model(algname, params):
    '''return a model specified by "method".'''
    if algname in ['SelfPacedEnsemble']:
        model = eval(algname)(base_estimator =params['base_estimator'], 
                     k_bins=params['k_bins'], 
                     n_estimators =params['n_estimators'])
    elif algname in ['GradientBoostingClassifier','RandomForestClassifier']:
        model = eval(algname)(n_estimators =params['n_estimators'])
    elif algname in METHODS:
        model = eval(algname)(base_estimator = params['base_estimator'], 
                     n_estimators =params['n_estimators'])
    elif algname =="ECUBoostRF":
        model = eval(algname)(L=params['n_estimators'],lamda=0.2,k=5,T=50)
    elif algname =="HashBasedUndersamplingEnsemble":
        model = eval(algname)(base_estimator= params['base_estimator'],n_iterations=params['n_estimators'])
    else:
        print(f'No such method support: {algname}')
    return model

def main(): 
    para = parse()
    algs=para.algorithm
    datasetname=para.dataset
    for alg in algs:
        ds_path,ds_name=os.path.split(datasetname)
        dataset_list=[]
        if os.path.isdir(datasetname):#is a set of data sets
            dataset_list=os.listdir(datasetname)
        else:#single data set
            dataset_list.append(ds_name)
        for dataset in tqdm(dataset_list):
            traintimes=[]
            testtime = []
            scores = [];
            X,y=dp.readDateSet(ds_path+'/'+dataset)
            sss = StratifiedShuffleSplit(n_splits=para.nfold, test_size=0.2,random_state=RANDOM_STATE)
            fold_cnt=0
            for train_index, test_index in sss.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                fold_cnt+=1
                params={'n_estimators':para.estimators, 'k_bins':10 ,'base_estimator':DecisionTreeClassifier()}
                model = init_model(
                        algname=alg,
                        params=params
                    )
                start_traintime = time.perf_counter()
                model.fit(X_train, y_train)
                traintimes.append(time.perf_counter()-start_traintime)
                start_testtime = time.perf_counter()
                y_pre=model.predict(X_test)
                testtime.append(time.perf_counter()-start_testtime)
                y_pred = model.predict_proba(X_test)[:, 1]#0 indicates the majority class，1 indicates the minority class
                y_pred[np.isnan(y_pred)] = 0
                metric=ImBinaryMetric(y_test,y_pre)
                scores.append([
                    metric.f1()
                    ,metric.MCC()
                    ,metric.aucroc(y_pred)
                ])
                
                del model
            print('ave_trainingrun_time:\t\t{:.3f}s'.format(np.mean(traintimes)))
            print('ave_testingrun_time:\t\t{:.3f}s'.format(np.mean(testtime)))
            print('------------------------------')
            print('Metrics:')
            df_scores = pd.DataFrame(scores, columns=['F1', 'MMC', 'AUC'])
            for metric in df_scores.columns.tolist():
                print ('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))
    return
if __name__ == '__main__':
    main()
            