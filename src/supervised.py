import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model, svm, preprocessing

x = pickle.load(open('../dat/X.pk'))
y = pickle.load(open('../dat/Y.pk'))
y = pd.merge(x,y,left_index=True,right_index=True)[['Age_At_Dx','Stage','Days_Survived']]


