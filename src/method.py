# -*- coding: utf-8 -*-
"""
Code for secure multi-label classification based on HE

@author: Jai Hyun Park (jhyunp@snu.ac.kr)
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import os
from IO import *
from sklearn.metrics import roc_auc_score
import sys, getopt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


######################
## Fixed Params     ##
######################

NUM_EPOCHS = 50
NUM_NODES = 64
DROPOUT_RATIO = 0.9
GOLDSCHMIDT_CONST = 64
GOLDSCHMIDT_ITER = 30


######################
## Some Functions   ##
######################

def loss_precision(x, d = 30):
    return x * (1 + (np.random.rand(1)[0] - 0.5) * 2 * 2**(-d))

def dist(x, y):
    return 1 - np.sum(x == y) / (len(x) * 2. - np.sum(x == y))

def our_exp(x):
    y = (np.array(x) + 16) / 80
    for i in range(4):
        y = y * y
        y = loss_precision(y)
    return y

def goldschmidt(x, M, n):
    base = 1 - np.sum(x) / M
    z = 1. / M
    for zeta in range(n):
        z *= 1 + base
        z = loss_precision(z)
        base *= base
        base = loss_precision(base)
    return z


######################
## Index Extraction ##
######################

def candidate1_generation(cn_thres, num_CN_gene, train_data):
    candidate1 = [0]
    for i in range(num_CN_gene):
        if dist(train_data[i], train_data[candidate1[-1]]) > cn_thres:
             candidate1 = np.concatenate((candidate1, [i]))
    return candidate1

def candidate2_generation(snpeff_thres, num_CN_gene, train_data, train_true):
    candidate2 = []
    for cancer in range(11):
        freq = train_data[num_CN_gene:, train_true == cancer].sum(axis = 1)
        candidate2 = np.concatenate((candidate2, freq.argsort()[-snpeff_thres:]))

    candidate2 = np.array(np.unique([int(i) for i in candidate2]))
    return candidate2


######################
## Shallow Network  ##
######################

def generate_model(train_data, train_label, val_data, val_label, batch_size = 32, epochs = 50, num_nodes = 128, dropout = 0.9, verbose = 0):
    train_label_encoded = tf.one_hot(train_label, 11)
    val_label_encoded = tf.one_hot(val_label, 11)
    
    model = tf.keras.models.Sequential()     
    model.add(tf.keras.layers.Dense(num_nodes, input_shape=(len(train_data[0]),), activation='linear', use_bias = False))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(11, activation=tf.nn.softmax, use_bias = False))

    model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(multi_label = True)])
    model.fit(train_data, train_label_encoded, batch_size=batch_size, epochs=epochs, validation_data=(val_data, val_label_encoded), verbose = verbose)

    return model

def model_generation(train_data, train_true, val_data, val_true, candidate):
    model = generate_model(train_data[candidate].T, train_true, val_data[candidate].T, val_true, epochs = NUM_EPOCHS, num_nodes = NUM_NODES, dropout = DROPOUT_RATIO, verbose = 0)
    w = pd.DataFrame(model.layers[0].get_weights()[0]).dot(pd.DataFrame(model.layers[2].get_weights()[0]))    
    return w


###########################
## Profiling on Test set ##
###########################

def micro_auc(val_true, val_pred, verbose = None, num_attr = 11):
    auc = roc_auc_score(tf.one_hot(val_true, num_attr), val_pred, average = 'micro')
    if verbose != None:
        print(str(verbose), auc)
    return auc

def profile(val_data, val_true, candidate, w):
    approx_accuracy, exact_accuracy, approx_auc, exact_auc = [0, 0, 0, 0]

    # Approx
    val_out = val_data[candidate].T.dot(w)
    val_expo = np.array([our_exp(x) for x in val_out])
    if np.max(np.sum(val_expo, axis=1)) < 2 / GOLDSCHMIDT_CONST:
        val_pred = np.array([x * goldschmidt(x, M=1/GOLDSCHMIDT_CONST, n=GOLDSCHMIDT_ITER) for x in val_expo])
        approx_auc = micro_auc(val_true, val_pred)

        val_res = np.argmax(val_pred, axis=1)
        approx_accuracy = (val_res == val_true).sum() / val_true.shape[0]

    # Exact
    val_expo = np.array([np.exp(x) for x in val_out])
    val_pred = np.array([x / np.sum(x) for x in val_expo])
    val_res = np.argmax(val_pred, axis=1)
    exact_auc = micro_auc(val_true, val_pred)
    exact_accuracy = (val_res == val_true).sum() / val_true.shape[0]

    return approx_accuracy, exact_accuracy, approx_auc, exact_auc



####################################################
#### Main
####################################################

if __name__ == '__main__':

    #############################################
    ## Params
    #############################################
    path = 'Jai Hyun Park'
    num_repeat = 10
    ct = 0.1
    st = 260

    argv = sys.argv
    try:
        opts, etc_args = getopt.getopt(argv[1:], 'h', ["cn=", "snpeff=", "repeat=", "path="])
    except getopt.GetoptError:
        print("invalid args")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("--cn"):
            ct = float(arg)
        elif opt in ("--snpeff"):
            st = int(arg)
        elif opt in ("--repeat"):
            num_repeat = int(arg)
        elif opt in ("--path"):
            path = arg

    print("* Experiment on ct =", ct, ", st =", st)
    print("* repeats", num_repeat, "times, and the results will be written in", path)


    #############################################
    ## Load data
    #############################################
    # Train data
    CN, SNPeff, true = read_all_data('data/Challenge/', '_challenge_CNs.txt', 'out/SNPeff/SNPeff_train.csv')
    full_data = np.concatenate([CN, SNPeff.T])
    num_CN_gene = CN.shape[0]

    # Test data
    CN_test, SNPeff_test, true_test = read_all_data('data/Evaluation/', '_evaluation_CNs.txt', 'out/SNPeff/SNPeff_test.csv')
    full_data_test = np.concatenate([CN_test, SNPeff_test.T])

    # gene list
    gene_list = np.array(pd.read_csv('out/SNPeff/variant_gene_list.csv', sep = '\t', header = None, dtype='str'))

    # shuffle
    train_data, train_true = random_shuffle(full_data, true)
    val_data, val_true = random_shuffle(full_data_test, true_test)


    #############################################
    ## Our method
    #############################################
    cand1 = candidate1_generation(ct, num_CN_gene, train_data)
    cand2 = candidate2_generation(st, num_CN_gene, train_data, train_true)
    candidate = np.concatenate((cand1, np.array(cand2 + num_CN_gene)))

    save_candidate1(cand1, ct, path)
    save_candidate2(cand2, gene_list, st, path)
     
    for ctr in range(num_repeat):
        w = model_generation(train_data, train_true, val_data, val_true, candidate)
        approx_acc, exact_acc, approx_auc, exact_auc = profile(val_data, val_true, candidate, w)
        print (ctr, ":", approx_acc, ",", exact_acc, ", ", approx_auc, ", ", exact_auc)
        save_w(w, ct, st, ctr, path)
