# -*- coding: utf-8 -*-
"""
Code for read, modify, and write CN/SNPeff/model data

@author: Jai Hyun Park (jhyunp@snu.ac.kr)
"""

import numpy as np
import pandas as pd
import os.path

LOW = 0.2
MODERATE = 0.5
MODIFIER = 0.9
HIGH = 1
cancer_list = ['Bladder', 'Breast', 'Bronchusandlung', 'Cervixuteri', 'Colon', 'Corpusuteri', 'Kidney', 'Liverandintrahepaticbileducts', 'Ovary', 'Skin', 'Stomach']
encodeSNPeff = ['MODIFIER', 'LOW', 'MODERATE', 'HIGH']


def random_shuffle(data, label):
    if data.shape[1] != len(label):
        print('the number of samples does not match!')
        return
    indices = np.arange(len(label))
    np.random.shuffle(indices)
    
    return data.T[indices].T, label[indices]


def read_CN(path, sep = '\t', header_row = True, header_column = True):
    if header_row:
        data = np.array(pd.read_csv(path, sep = sep, header = 0))
    else:
        data = np.array(pd.read_csv(path, sep = sep, header = None))
    if header_column:
        data = data[:, 1:]
    return data.astype(np.float32)


def read_CNs(path_head, path_tail):
    data = []
    label = []
    for cancer in cancer_list:
        data.append(read_CN(path_head + cancer + path_tail, sep = '\t', header_row = True, header_column = True))
        label += [cancer_list.index(cancer)] * data[-1].shape[1]

    data = np.hstack(data)
    label = np.array(label)
    return data, label


def read_variants(path_head, path_tail, path_out, path_gene_list):
    data = []    
    for cancer in cancer_list:
        data.append(np.array(pd.read_csv(path_head + cancer + path_tail, sep = '\t', header = None)))
    data = np.vstack(data)

    fexist = os.path.isfile(path_gene_list)
    if fexist:
        gene_list = list(np.array(pd.read_csv(path_gene_list, sep = '\t', header = None, dtype = 'str')).flatten())
    else:
        gene_list = list(set(data[:, 1].T))
        gene_list.sort()
    num_genes = len(gene_list)

    SNPeffs = []
    name = 'Jai Hyun Park'
    idx = -1
    for datum in data:
        if datum[0] != name:
            if name != 'Jai Hyun Park':
                SNPeffs.append(SNPeff)
            SNPeff = np.zeros(num_genes)
            name = datum[0]
            idx += 1

        if not fexist or datum[1] in gene_list:
            SNPeff[gene_list.index(datum[1])] = encodeSNPeff.index(datum[7]) + 1
        else:
            print('gene not in list', datum[1])

    SNPeffs.append(SNPeff)
    SNPeffs = np.vstack(SNPeffs)
    
    pd.DataFrame(SNPeffs).to_csv(path_out, sep = '\t', header = None, index = False)
    if not os.path.isfile(path_gene_list):
        pd.DataFrame(gene_list).to_csv(path_gene_list, sep = '\t', header = None, index = False)

    return gene_list, SNPeffs


def read_variants_csv(path_csv):
    SNPeff = np.array(pd.read_csv(path_csv, sep = '\t', header = None, dtype=float))
    SNPeff[SNPeff == 1] = MODIFIER
    SNPeff[SNPeff == 2] = LOW
    SNPeff[SNPeff == 3] = MODERATE
    SNPeff[SNPeff == 4] = HIGH
    return SNPeff


def read_all_data(cn_head, cn_tail, snpeff_csv):
    CN, true = read_CNs(cn_head, cn_tail)
    SNPeff = read_variants_csv(snpeff_csv)
    return CN, SNPeff, true


def save_candidate1(candidate1, cn_thres, path):
    pd.DataFrame(candidate1).to_csv(path+'cand1_cn' +  str(cn_thres) + '.csv', sep = ',', header = None, index = False)


def save_candidate2(candidate2, gene_list, snpeff_thres, path):
    pd.DataFrame([gene_list[i] for i in candidate2]).to_csv(path + 'cand2_snpeff' +  str(snpeff_thres) + '.csv', sep = ',', header = None, index = False)


def save_w(w, cn_thres, snpeff_thres, ctr, path):
    pd.DataFrame(w).to_csv(path + 'w_cn' +  str(cn_thres) + '_snpeff' + str(snpeff_thres) + '_ctr' + str(ctr) + '.csv', sep = ',', header = None, index = False)


if __name__ == '__main__':
    read_variants(path_head = 'data/Challenge/', path_tail = '_challenge_variants.txt', path_out = 'out/SNPeff/SNPeff_train.csv', path_gene_list = 'out/SNPeff/variant_gene_list.csv')
    read_variants(path_head = 'data/Evaluation/', path_tail = '_evaluation_variants.txt', path_out = 'out/SNPeff/SNPeff_test.csv', path_gene_list = 'out/SNPeff/variant_gene_list.csv')


