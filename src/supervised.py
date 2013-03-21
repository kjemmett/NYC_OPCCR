import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing, cross_validation, linear_model, svm, metrics

x = pickle.load(open('../dat/X.pk'))
y = pickle.load(open('../dat/Y.pk'))
y = pd.merge(x,y,left_index=True,right_index=True)[['Age_At_Dx','Stage','Days_Survived']]

x_scaled = preprocessing.scale(np.array(x))
stage = np.zeros((263,))
survival = np.zeros((263,))
counter = 0
for elem in y['Stage']:
    if elem == 'I':
        stage[counter] = 1
    elif elem == 'II':
        stage[counter] = 2
    elif elem == 'III':
        stage[counter] = 3
    else:
        stage[counter] = 4
    counter += 1
counter = 0
for elem in y['Days_Survived']:
    survival[counter] = elem
    counter += 1

kfold = cross_validation.KFold(len(x_scaled), n_folds=10)

##linear regression with regularization 
linreg = linear_model.Lasso(alpha=10)
#print cross_validation.cross_val_score(linreg, x_scaled, survival, cv=kfold, n_jobs=-1)

##support vector regression
svr = svm.SVR()
svr.fit(x_scaled, survival)
print metrics.mean_squared_error(stage, svr.predict(x_scaled))
print metrics.r2_score(stage, svr.predict(x_scaled))
#print cross_validation.cross_val_score(svr, x_scaled, survival, cv=kfold, n_jobs=-1)

##logistic regression
logreg = linear_model.LogisticRegression()
logreg.fit(x_scaled, stage)
print logreg.score(x_scaled, stage)
print metrics.confusion_matrix(stage, logreg.predict(x_scaled))

#support vector classification
#svc = svm.LinearSVC(penalty='l1',dual=False)
#svc.fit(x_scaled, stage)
#print svc.score(x_scaled, stage)
#print metrics.confusion_matrix(stage, svc.predict(x_scaled))
