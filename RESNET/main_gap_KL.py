# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:02:17 2022

@author: Joker
"""
from utils.input_data import read_data_sets
import utils.datasets as ds
import utils.augmentation as augm
import utils.helper as hlp
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from main_functions import *
from resnet import Classifier_RESNET
import numpy as np
import os
from os import listdir
import pandas as pd
from imbalance_degree import imbalance_degree
import random as random
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix




bench = list()

#balance = pd.read_csv('balance_measure.csv') #We use Shanon Entropy as reference

folders = os.listdir('data')


for i in range(len(folders)):
    tmp_bench = list()
    dataset = folders[i]
    
    print(dataset)
    print(f'{i}/{len(folders)}')
    
    #load data
    nb_class = ds.nb_classes(dataset)
    nb_dims = ds.nb_dims(dataset)
    train_data_file = os.path.join("data/", dataset, "%s_TRAIN.tsv"%dataset)
    test_data_file = os.path.join("data/", dataset, "%s_TEST.tsv"%dataset)
    
    x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter="\t")
    
    y_train = ds.class_offset(y_train, dataset)
    y_test= ds.class_offset(y_test, dataset)
    nb_timesteps = int(x_train.shape[1] / nb_dims)
    input_shape = (nb_timesteps , nb_dims)
    
    x_train_max = np.max(x_train)
    x_train_min = np.min(x_train)
    #normalise in [-1;1]
    x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
    x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.
    
    nb_timesteps = int(x_train.shape[1] / nb_dims)
    input_shape = (nb_timesteps , nb_dims)
    x_test = x_test.reshape((-1, input_shape[0], input_shape[1])) 
    x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))
    y_test = to_categorical(ds.class_offset(y_test, dataset), nb_class) 
    
   
    _, rat = np.unique(y_train, return_counts=True)
    y_majority_idxs = list()
    majority_class = np.argmax(rat)
    for label in range(len(y_train)):
      if (rat[int(y_train[label])] == rat[majority_class]):
        y_majority_idxs.append(label)


    y_majority_idxs = np.array(y_majority_idxs)
    print(y_majority_idxs)
    y_non_majority_idxs = np.where(np.array(rat) != rat[majority_class])[0].tolist()


    x_train_new = x_train[y_majority_idxs]
    y_train_new = y_train[y_majority_idxs]

    x_mino,y_mino = take_sample(x_train,y_train, 2, y_non_majority_idxs)

    x_train_new = np.concatenate((x_train_new, np.array(x_mino)))

    y_train_new = np.concatenate((y_train_new, y_mino))
    #
    #tmp_raw=0
    #for ite in range(2):
    #    raw = raw_data(dataset, x_train, y_train, x_test, y_test, input_shape, nb_class)
    #    tmp_raw+=raw
    #tmp_bench.append(tmp_raw/2)
    #tmp_ros = 0
    #for ite in range(2):

     #   ros = ROS_test(dataset,x_train, y_train, x_test, y_test, input_shape,  nb_class)
     #   tmp_ros+=ros

    #tmp_bench.append(tmp_ros/2)
    #tmp_jit = 0
    #for i in range(2):

    #    jit = jitter_test(dataset, x_train, y_train, x_test,  y_test, input_shape,  nb_class, rat, majority_class)
     #   tmp_jit+= jit
    #tmp_bench.append(tmp_jit/2)

    #tmp_tw = 0

    #for i in range(2):

     #   tw = tw_test(dataset, x_train, y_train, x_test,  y_test, input_shape,  nb_class, rat, majority_class)
      #  tmp_tw+=tw
    #tmp_bench.append(tmp_tw/2)

    

    evolutionf = list() #f1 scores
    evolutiong = list() #g means
    evolutiona = list() #accuracy
    evolutionmcc = list() #mcc
    evolutionrec = list() #precision
    evolutionpres = list() #recall

  
    copy = y_train_new.tolist()
    print(copy)
    rat_test = np.array(rat)
    tmp = list() # list for ID
    res = list() # list for accu
    e = 1/nb_class

    accu = 0
    mcc = 0
    f = np.array([0.0 for cl in range(nb_class)])
    rec = np.array([0.0 for cl in range(nb_class)])
    pres = np.array([0.0 for cl in range(nb_class)])
    g = np.array([0.0 for cl in range(nb_class)])

    for i in range(3):
        taccu,tmcc, tf, trec, tpres, tg = SMOTE_test(dataset, x_train_new, y_train_new, x_test,  np.argmax(y_test, axis = 1), input_shape,  nb_class)
        
        accu += taccu
        mcc += tmcc 
        f += tf 
        
        rec += trec 
        pres += tpres  
        g += tg 

    evolutionf.append(f/3) # f1 scores
    evolutiong.append(g/3) #g means
    evolutiona.append(accu/3) #accuracy
    evolutionmcc.append(mcc/3) #mcc
    evolutionrec.append(rec/3) #precision
    evolutionpres.append(pres/3) #recall
    ee = [rat[majority_class] for ouiclass in range(nb_class)]

    _, new_rat = np.unique(copy, return_counts=True)
    

    res.append(accu)
    
    tmp.append(KL(ee, new_rat))


    total_add_nb = 0
    for x in new_rat:
      total_add_nb += new_rat[majority_class] - x
    print(total_add_nb)
    add_nb = 0
    #Sorted class
    sorted_rat_next = np.argsort(new_rat)
    



    #Trig is one hot encodding [0,0,0,...,1,...,0], if stop[i] = 1 => we cannot add new data with label i without adding a new minority class
    while(add_nb != total_add_nb):

      for i in sorted_rat_next: 
        
        if (new_rat[i] != new_rat[majority_class]):
          
          copy.append(i)
          _, tmp_rat = np.unique(copy, return_counts=True)
          
          
          
          
          tmp.append(KL(ee,tmp_rat))
          print(tmp_rat)
          sp_strg = {i:tmp_rat[i] for i in range(len(tmp_rat))}
          accu = 0
          mcc = 0
          f  = np.array([0.0 for cl in range(nb_class)])
          rec = np.array([0.0 for cl in range(nb_class)])
          pres = np.array([0.0 for cl in range(nb_class)])
          g = np.array([0.0 for cl in range(nb_class)])
          


          for ite in range(3):
            taccu,tmcc, tf, trec, tpres, tg = SMOTE_test(dataset, x_train_new, y_train_new, x_test,  np.argmax(y_test, axis = 1), input_shape,  nb_class, sp_strg)
            

            accu += taccu
            mcc += tmcc 
            f += tf 
            
            rec += trec 
            pres += tpres  
            g += tg 

          

          evolutionf.append(f/3) # f1 scores
          evolutiong.append(g/3) #g means
          evolutiona.append(accu/3) #accuracy
          evolutionmcc.append(mcc/3) #mcc
          evolutionrec.append(rec/3) #precision
          evolutionpres.append(pres/3) #recall
          add_nb +=1
            
            

        

    
    os.makedirs('Results/'+ dataset + '/SMOTE', exist_ok=True)
    
    
    KL = pd.DataFrame(tmp)

    pd.DataFrame(evolutionf).to_csv('Results/'+ dataset + '/SMOTE/' + 'f1_evolution.csv')
    pd.DataFrame(evolutiong).to_csv('Results/'+ dataset + '/SMOTE/' + 'g_evolution.csv')
    pd.DataFrame(evolutiona).to_csv('Results/'+ dataset + '/SMOTE/' + 'accu_evolution.csv')
    pd.DataFrame(evolutionmcc).to_csv('Results/'+ dataset + '/SMOTE/' + 'mcc_evolution.csv')
    pd.DataFrame(evolutionrec).to_csv('Results/'+ dataset + '/SMOTE/' + 'rec_evolution.csv')
    pd.DataFrame(evolutionpres).to_csv('Results/'+ dataset + '/SMOTE/' + 'pres_evolution.csv')
    KL.to_csv(f'Results/{dataset}/SMOTE/KL.csv')
