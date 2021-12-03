# -*- coding: utf-8 -*-
"""
Code for a table of the model size for each parameter.

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

argv = sys.argv
try:
    opts, etc_args = getopt.getopt(argv[1:], 'h', ["path=", "cn-start=", "cn-step=", "cn-stop=", "snpeff-start=", "snpeff-step=", "snpeff-stop="])
except getopt.GetoptError:
    print("invalid args")
    sys.exit(2)

for opt, arg in opts:
    if opt in ("--path"):
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
print("* the results will be written in", path, "num_cand.csv")


#############################################
## Load data
#############################################
# Train data
CN, SNPeff, true = read_all_data('data/Challenge/', '_challenge_CNs.txt', 'out/SNPeff/SNPeff_train.csv')
full_data = np.concatenate([CN, SNPeff.T])
num_CN_gene = CN.shape[0]


#############################################
## Our method
#############################################
num_cand1s = [len(candidate1_generation(ct, num_CN_gene, full_data)) for ct in cts]
num_cand2s = [len(candidate2_generation(st, num_CN_gene, full_data, true)) for st in sts]
num_cands = [[n1 + n2 for n2 in num_cand2s] for n1 in num_cand1s]

pd.DataFrame(num_cands, index=cts).to_csv(path + 'num_cands.csv', sep = ',', header = sts, index = True)

