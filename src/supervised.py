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
linreg = linear_model.LinearRegression()
lasso = linear_model.Lasso(alpha=10)
print 'Linear Regression R^2 Across CV Folds:'
print cross_validation.cross_val_score(linreg, x_scaled, survival, cv=kfold, n_jobs=-1)
print 'Lasso R^2 Across CV Folds:'
print cross_validation.cross_val_score(lasso, x_scaled, survival, cv=kfold, n_jobs=-1)

##logistic regression
logreg = linear_model.LogisticRegression()
logreg.fit(x_scaled, stage)
print 'Logistic Score:', logreg.score(x_scaled, stage)
print 'Confusion Matrix:'
print metrics.confusion_matrix(stage, logreg.predict(x_scaled))

##support vector classification
svc = svm.LinearSVC(penalty='l1',dual=False)
svc.fit(x_scaled, stage)
print 'SVC Score:', svc.score(x_scaled, stage)
print 'Confusion Matrix:'
print metrics.confusion_matrix(stage, svc.predict(x_scaled))
