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

# gene list
gene_list = np.array(pd.read_csv('out/SNPeff/variant_gene_list.csv', sep = '\t', header = None, dtype='str'))



#############################################
## Our method
#############################################
approx_aucs = [[0. for st in sts] for ct in cts]
exact_aucs = [[0. for st in sts] for ct in cts]
approx_accs = [[0. for st in sts] for ct in cts]
exact_accs = [[0. for st in sts] for ct in cts]
# ctrs = [[0. for st in sts] for ct in cts]


full_num = full_data.shape[1]
val_size = (int) (full_num / num_repeat)
print(full_num, val_size)

for idx_ct in range(len(cts)):
    for idx_st in range(len(sts)):

        full_data, true = random_shuffle(full_data, true)

        print("ct:", cts[idx_ct], ", st:", sts[idx_st])
        
        ctr = 0
        exact_auc, approx_auc = [0, 0]
        exact_acc, approx_acc = [0, 0]

        for i in range(num_repeat):
            
            if (i > 0):
                full_data = np.roll(full_data, shift=val_size, axis=1)
                true = np.roll(true, shift=val_size, axis=0)

            train_data, train_true = full_data[:, val_size:], true[val_size:]
            val_data, val_true = full_data[:, :val_size], true[:val_size]

            cand1 = candidate1_generation(cts[idx_ct], num_CN_gene, train_data)
            cand2 = candidate2_generation(sts[idx_st], num_CN_gene, train_data, train_true)
            candidate = np.concatenate((cand1, np.array(cand2) + num_CN_gene))

            w = model_generation(train_data, train_true, val_data, val_true, candidate)
            tmp_approx_acc, tmp_exact_acc, tmp_approx_auc, tmp_exact_auc = profile(val_data, val_true, candidate, w)
            
            approx_auc += tmp_approx_auc
            exact_auc += tmp_exact_auc
            approx_acc += tmp_approx_acc
            exact_acc += tmp_exact_acc
            
            if tmp_approx_auc != 0:
                ctr += 1

        if ctr != 0:
            approx_aucs[idx_ct][idx_st], exact_aucs[idx_ct][idx_st] = [approx_auc / ctr, exact_auc / num_repeat]
            approx_accs[idx_ct][idx_st], exact_accs[idx_ct][idx_st] = [approx_acc / ctr, exact_acc / num_repeat]
        else:
            approx_aucs[idx_ct][idx_st], exact_aucs[idx_ct][idx_st] = [0, exact_auc / num_repeat]
            approx_accs[idx_ct][idx_st], exact_accs[idx_ct][idx_st] = [0, exact_acc / num_repeat]

        print("acc(approx/exact):", approx_accs[idx_ct][idx_st], exact_accs[idx_ct][idx_st])
        print("auc(approx/exact):", approx_aucs[idx_ct][idx_st], exact_aucs[idx_ct][idx_st])

        pd.DataFrame(approx_aucs, index=cts).to_csv(path + 'approx_aucs.csv', sep = ',', header = sts, index = True)
        pd.DataFrame(exact_aucs, index=cts).to_csv(path + 'exact_aucs.csv', sep = ',', header = sts, index = True)
        pd.DataFrame(approx_accs, index=cts).to_csv(path + 'approx_accs.csv', sep = ',', header = sts, index = True)
        pd.DataFrame(exact_accs, index=cts).to_csv(path + 'exact_accs.csv', sep = ',', header = sts, index = True)

