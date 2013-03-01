import string
import subprocess
import numpy as np
from scipy.stats import t
from glob import glob
from math import log
import collections
import pickle

epsilon = 10**(-10)

def get_barcode(x):
    return x.split('__')[2][:12]

stats_control = np.zeros((20532,2))
stats_tumor = np.zeros((20532,5))
gene_symbols = []

expression_control_dir = '/Users/sakellarios/Documents/04-MD-PhD/5-Research/TCGA/hnscc/expression_control/'
expression_tumor_dir = '/Users/sakellarios/Documents/04-MD-PhD/5-Research/TCGA/hnscc/expression_tumor/'

#VERIFY THAT EACH TUMOR EXPRESSION PROFILE ALSO HAS CLINCAL DATA
#GENERATE LIST OF BARCODES FOR WHICH THERE ARE DUPLICATE EXPRESSION PROFILES
clinical_barcodes = subprocess.check_output('cut -f 1 ../dat/y_data.txt',shell=True).split()[1:]
dir_contents = subprocess.check_output('ls', cwd=expression_tumor_dir).split()
dir_genes_only = []
for elem in dir_contents:
    if elem.find('gene') >= 0:
        dir_genes_only.append(elem)
dir_barcodes = [get_barcode(fname) for fname in dir_genes_only]
intersect_list = [fname for fname in dir_barcodes if fname in clinical_barcodes]
assert(len(intersect_list) == len(dir_barcodes))
duplicate_barcodes = [x for x,y in collections.Counter(dir_barcodes).items() if y > 1]

#GENERATE THE LIST OF GENE SYMBOLS USING ONE ARBITRARY FILE
f_handle = open('/Users/sakellarios/Documents/04-MD-PhD/5-Research/TCGA/hnscc/expression_control/unc.edu__IlluminaHiSeq_RNASeq__TCGA-CV-6934-11A-01R-1915-07__expression_gene.txt','r')
f_handle.readline() #burn a line
for line in f_handle:
    line_elements = line.split()
    gene_symbols.append(line_elements[1])
f_handle.close()

#GENERATE CONTROL STATISTICS ON LOG TRANSFORMED RPKM
gene_files = []
for f in glob(expression_control_dir + '*'):
    if f.find('gene.txt') >= 0:
        gene_files.append(f)

n_c = 0
for f in gene_files:
    f_handle = open(f)
    f_handle.readline()
    n_c = n_c + 1
    row = 0
    for line in f_handle:
        columns = line.split()
        delta = log(float(columns[4])+epsilon) - stats_control[row][0]
        R = delta * 1 / n_c
        stats_control[row][0] = stats_control[row][0] + R
        M2 = stats_control[row][1] + delta * R * (n_c - 1)
        stats_control[row][1] = M2
        row = row + 1
    f_handle.close()
stats_control[:,1] = stats_control[:,1] / (n_c - 1) #turn M2 into variances

#GENERATE TUMOR STATISTICS ON LOG TRANSFORMED RPKM
gene_files = []
for f in glob(expression_tumor_dir + '*'): #redundant with earlier subprocess code
    if f.find('gene.txt') >= 0:
        gene_files.append(f)

n_t = 0
for f in gene_files:
    weight = 1
    b_code = get_barcode(f)
    if b_code in duplicate_barcodes:
        weight = 0.5
    f_handle = open(f)
    f_handle.readline()
    n_t = n_t + weight
    row = 0
    for line in f_handle:
        columns = line.split()
        delta = log(float(columns[4])+epsilon) - stats_tumor[row][0]
        R = delta * weight / n_t
        stats_tumor[row][0] = stats_tumor[row][0] + R
        M2 = stats_tumor[row][1] + delta * R * (n_t - weight)
        stats_tumor[row][1] = M2
        row = row + 1
    f_handle.close()
stats_tumor[:,1] = stats_tumor[:,1] / (n_t - 1) #turn M2 into variances

#LOOP THRU STATS AND PICK FEATURES
#THRESHOLD WELCH t-STATISTIC and THRESHOLD TUMOR VARIANCE
for row in range(len(stats_tumor)):
    m_t = stats_tumor[row][0]
    v_t = stats_tumor[row][1]
    m_c = stats_control[row][0]
    v_c = stats_control[row][1]
    #t_welch = (m_t - m_c) / np.sqrt(v_t / n_t + v_c / n_c)
    #df_welch = (v_t / n_t + v_c / n_c)**2 / (v_t**2 / (n_t**2 * (n_t - 1)) + v_c**2 / (n_c**2 * (n_c - 1)))
    df_non_welch = n_t + n_c - 2
    t_non_welch = (m_t - m_c) / (np.sqrt(((n_t - 1) * v_t + (n_c - 1) * v_c) / df_non_welch) * np.sqrt(1.0/n_t + 1.0/n_c))
    rv = t(df_non_welch)
    p_value = rv.sf(abs(t_non_welch)) * 2
    stats_tumor[row][2] = t_non_welch
    stats_tumor[row][3] = df_non_welch
    stats_tumor[row][4] = p_value

f_out_tumor = open('../out/features_tumor.pk','w')
f_out_control = open('../out/features_control.pk','w')
pickle.dump(stats_tumor,f_out_tumor)
pickle.dump(stats_control,f_out_control)

#COMPUTE THE MOST SIGNIFICANT GENES BY P_VALUE & TUMOR VARIANCE
def get_sig_genes(thresh_var, thresh_pval):
    mask = np.bitwise_and(stats_tumor[:,4] < 10**(-thresh_pval), stats_tumor[:,1] > thresh_var)
    indices = mask.nonzero()[0]
    feats = []
    for i in indices:
        feats.append(gene_symbols[i])
    return feats

feature_set = get_sig_genes(5,10)

#LOOP OVER RAW DATA, CHECKING FOR DUPLICATE BAR CODE, AND WRITING TO ../dat THE FINAL FEATURE SPACE
dup = False
tmp = []
for f in gene_files:
    b_code = get_barcode(f)
    f_out = open('../dat/aberrant_genes_%s.txt' %(b_code),'w')
    f_in = open(f,'r')
    f_in.readline()
    if dup == False and b_code in duplicate_barcodes:
        dup = True
        tmp = f_in.read().split('\n')[:-1]
        continue
    elif dup == True and b_code in duplicate_barcodes:
        dup = False
        counter = 0
        for line in f_in:    
            columns = line.split()
            if columns[1] in feature_set:
                gene_symbol = columns[1]
                rpkm = str(0.5 * float(columns[4]) + 0.5 * float(tmp[counter].split('\t')[4]))
                f_out.write(gene_symbol + '\t' + rpkm + '\n')
            counter = counter + 1
    else:
        for line in f_in:    
            columns = line.split()
            if columns[1] in feature_set:
                gene_symbol = columns[1]
                rpkm = columns[4]
                f_out.write(gene_symbol + '\t' + rpkm + '\n')

