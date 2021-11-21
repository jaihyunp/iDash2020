# -*- coding: utf-8 -*-
"""
Code for generating a result tables

@author: Jai Hyun Park (jhyunp@snu.ac.kr)
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import os
from IO import *
from method import *
from sklearn.metrics import roc_auc_score
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys, getopt


#############################################
## Params
#############################################

cn_start, cn_step, cn_stop = [0., 0., 0.]
snpeff_start, snpeff_step, snpeff_stop = [0, 0, 0]

path = 'Jai Hyun Park'
num_repeat = 10

argv = sys.argv
try:
    opts, etc_args = getopt.getopt(argv[1:], 'h', ["repeat=", "path=", "cn-start=", "cn-step=", "cn-stop=", "snpeff-start=", "snpeff-step=", "snpeff-stop="])
except getopt.GetoptError:
    print("invalid args")
    sys.exit(2)

for opt, arg in opts:
    if opt in ("--repeat"):
        num_repeat = int(arg)
    elif opt in ("--path"):
        path = arg
    elif opt in ("--cn-start"):
        cn_start = float(arg)
    elif opt in ("--cn-step"):
        cn_step = float(arg)
    elif opt in ("--cn-stop"):
        cn_stop = float(arg)
    elif opt in ("--snpeff-start"):
        snpeff_start = int(arg)
    elif opt in ("--snpeff-step"):
        snpeff_step = int(arg)
    elif opt in ("--snpeff-stop"):
        snpeff_stop = int(arg)


cts = np.arange(cn_start, cn_stop, cn_step) #Cn Threshold
sts = np.arange(snpeff_start, snpeff_stop, snpeff_step) #Snpeff Threshold

print("* Experiment on ct = (", cn_start, ":", cn_step, ":", cn_stop, "), st = (", snpeff_start, ":", snpeff_step, ":", snpeff_stop, ")")
print("* repeat ", num_repeat, "times, and the results will be written in", path)



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

cand1s = [candidate1_generation(ct, num_CN_gene, train_data) for ct in cts]
cand2s = [candidate2_generation(st, num_CN_gene, train_data, train_true) for st in sts]

num_cands = [[0. for st in sts] for ct in cts]
approx_aucs = [[0. for st in sts] for ct in cts]
exact_aucs = [[0. for st in sts] for ct in cts]
approx_accs = [[0. for st in sts] for ct in cts]
exact_accs = [[0. for st in sts] for ct in cts]
ctrs = [[0. for st in sts] for ct in cts]

for idx_ct in range(len(cts)):
    for idx_st in range(len(sts)):

        print("ct:", cts[idx_ct], ", st:", sts[idx_st])
        
        candidate = np.concatenate((cand1s[idx_ct], np.array(cand2s[idx_st]) + num_CN_gene))
        num_cand = len(candidate)

        ctr = 0
        exact_auc, approx_auc = [0, 0]
        exact_acc, approx_acc = [0, 0]
        for i in range(num_repeat):
            w = model_generation(train_data, train_true, val_data, val_true, candidate)
            tmp_approx_acc, tmp_exact_acc, tmp_approx_auc, tmp_exact_auc = profile(val_data, val_true, candidate, w)
            
            approx_auc += tmp_approx_auc
            exact_auc += tmp_exact_auc
            approx_acc += tmp_approx_acc
            exact_acc += tmp_exact_acc
            
            if tmp_approx_auc != 0:
                ctr += 1

        num_cands[idx_ct][idx_st] = num_cand
        ctrs[idx_ct][idx_st] = ctr

        if ctr != 0:
            approx_aucs[idx_ct][idx_st], exact_aucs[idx_ct][idx_st] = [approx_auc / ctr, exact_auc / num_repeat]
            approx_accs[idx_ct][idx_st], exact_accs[idx_ct][idx_st] = [approx_acc / ctr, exact_acc / num_repeat]
        else:
            approx_aucs[idx_ct][idx_st], exact_aucs[idx_ct][idx_st] = [0, exact_auc / num_repeat]
            approx_accs[idx_ct][idx_st], exact_accs[idx_ct][idx_st] = [0, exact_acc / num_repeat]

        print("acc(approx/exact):", approx_accs[idx_ct][idx_st], exact_accs[idx_ct][idx_st])
        print("auc(approx/exact):", approx_aucs[idx_ct][idx_st], exact_aucs[idx_ct][idx_st])

        pd.DataFrame(num_cands, index=cts).to_csv(path + 'num_cands.csv', sep = ',', header = sts, index = True)
        pd.DataFrame(approx_aucs, index=cts).to_csv(path + 'approx_aucs.csv', sep = ',', header = sts, index = True)
        pd.DataFrame(exact_aucs, index=cts).to_csv(path + 'exact_aucs.csv', sep = ',', header = sts, index = True)
        pd.DataFrame(approx_accs, index=cts).to_csv(path + 'approx_accs.csv', sep = ',', header = sts, index = True)
        pd.DataFrame(exact_accs, index=cts).to_csv(path + 'exact_accs.csv', sep = ',', header = sts, index = True)
        pd.DataFrame(ctrs, index=cts).to_csv(path + 'ctrs.csv', sep = ',', header = sts, index = True)

