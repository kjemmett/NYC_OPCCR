import matplotlib.pyplot as plt, matplotlib as mpl, pandas as pd, numpy as np, pickle as pk
from mpl_toolkits.mplot3d import Axes3D

x = pk.load(open('../dat/X.pk'))
y = pk.load(open('../dat/Y.pk'))
y = pd.merge(x,y,left_index=True,right_index=True)[['Age_At_Dx','Stage','Days_Survived']]

#Scatter plot of Days_Survived vs. Age_At_Dx
fig1 = plt.scatter(y['Age_At_Dx'], y['Days_Survived'])
plt.title('Survival vs. Age')
plt.xlabel('Age at Diagnosis (years)')
plt.ylabel('Survival (days)')
plt.xlim(0,10*np.max(y['Age_At_Dx'])/10+10)
plt.ylim(0,1000*np.max(y['Days_Survived'])/1000+1000)
plt.xticks(10*np.arange(10))
plt.savefig('../doc/surv_vs_age.png')
plt.show()

#Histogram of Days_Survived
fig2 = plt.hist(y['Days_Survived'], bins = 500 * np.arange(0, np.max(y['Days_Survived'])/500 + 1))
plt.title('Survival')
plt.xlabel('Survival (days)')
plt.ylabel('Frequency')
plt.savefig('../doc/hist_surv.png')
plt.show()

#boxplot of Age_At_Dx by Stage
stage_IV_diagnosis = pd.concat([y['Age_At_Dx'][y['Stage']=='IVA'],y['Age_At_Dx'][y['Stage']=='IVB']]) #concatenates stages IVA and IVB
box_diagnosis = [y['Age_At_Dx'][y['Stage']=='I'], y['Age_At_Dx'][y['Stage']=='II'],y['Age_At_Dx'][y['Stage']=='III'],stage_IV_diagnosis]
fig3 = plt.boxplot(box_diagnosis)
plt.title('Age at Diagnosis by Stage')
plt.xlabel('Stage')
plt.ylabel('Age at Diagnosis (years)')
plt.savefig('../doc/box_age_by_stage.png')
plt.show()

#boxplot of Days_Survived by Stage
box_IV_survival = pd.concat([y['Days_Survived'][y['Stage']=='IVA'],y['Days_Survived'][y['Stage']=='IVB']]) #concatenates stages IVA and IVB
box_survival = [y['Days_Survived'][y['Stage']=='I'], y['Days_Survived'][y['Stage']=='II'],y['Days_Survived'][y['Stage']=='III'],box_IV_survival]
fig4 = plt.boxplot(box_survival)
plt.title('Survival by Stage')
plt.xlabel('Stage')
plt.ylabel('Survival (days)')
plt.savefig('../doc/box_surv_by_stage.png')
plt.show()

#correlation matrix heatmap
r = x.corr()
mask = np.tri(r.shape[0], k = -1)
r = np.ma.array(r, mask = mask)
cmap = mpl.cm.get_cmap('jet',10)
cmap.set_bad('w')
fig5 = plt.matshow(r)
plt.title('Corrleation Matrix Heatmap')
plt.savefig('../doc/corr_heatmap.png')
plt.show()

#principal component analysis
pca = mpl.mlab.PCA(x)

#(1)curve of cumulative unexplained variance
cum = np.zeros(x.shape[1],)
for i in range(x.shape[1]):
    cum[i] = reduce(lambda x,y: x+y, pca.fracs[:i+1])
fig6 = plt.figure()
ax1 = fig6.add_subplot(121)
ax1 = plt.plot(np.arange(1,x.shape[1]+1), 1-cum)
plt.title('Unexplained Variance')
plt.xlabel('Number of Factors')
plt.ylabel('Unexplained Variance')

#(2)3d scatter of Days_Survived by Stage along 3 principal axes
dp = []
arr = np.array(x)
for i in range(np.shape(arr)[0]):
    dp.append(pca.project(arr[i,:], minfrac = pca.fracs[2]))
dp = np.array(dp)
ax2 = fig6.add_subplot(122, projection = '3d')
for c, stage in zip("rgby", ['I','II','III','IVA']):
    ax2.scatter(dp[y['Stage'] == stage, 0], dp[y['Stage'] == stage,1], dp[y['Stage'] == stage, 2], c=c)
plt.title('Principal Component Analysis')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig('../doc/pca.png')
plt.show()
