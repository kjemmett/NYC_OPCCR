import numpy as np
import pandas as pd
import pickle
from glob import glob

f_age = open('../dat/y_age_at_dx.txt','r')
f_stage = open('../dat/y_staging.txt','r')
f_surv = open('../dat/y_survival.txt','r')

y_age = pd.read_csv(f_age,sep='\t',header=None,index_col=0,names=['Age_At_Dx'])
y_stage = pd.read_csv(f_stage,sep='\t',header=None,index_col=0,names=['Stage'])
y_surv = pd.read_csv(f_surv,sep='\t',header=None,index_col=0,names=['Days_Survived'])
y_total = pd.merge(y_age,y_stage,left_index=True,right_index=True)
y_total = pd.merge(y_total,y_surv,left_index=True,right_index=True)

f_out_y = open('../dat/Y.pk','w')
pickle.dump(y_total,f_out_y)

txt_files = glob('../dat/aberrant*')

for fname in txt_files:
    barcode = fname[22:34]
    f_handle = open(fname,'r')
    x_new = pd.read_csv(f_handle,sep='\t',header=None,index_col=0,names=[barcode])
    try:
        x_total = pd.merge(x_total,x_new,left_index=True,right_index=True)
    except:
        x_total = x_new

f_out_x = open('../dat/X.pk','w')
pickle.dump(x_total.transpose(), f_out_x)


