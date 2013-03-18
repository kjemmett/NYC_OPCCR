import matplotlib.pyplot as plt, matplotlib.mlab as ml, pandas as pd, numpy as np, pickle as pk
from mpl_toolkits.mplot3d import Axes3D

x = pk.load(open('../dat/X.pk'))
y = pk.load(open('../dat/Y.pk'))
y = pd.merge(x,y,left_index=True,right_index=True)[['Age_At_Dx','Stage','Days_Survived']]

pca = ml.PCA(x)
dp = []
arr = np.array(x)
for i in range(np.shape(arr)[0]):
    dp.append(pca.project(arr[i,:], minfrac = pca.fracs[2]))
dp = np.array(dp)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
for c, stage in zip("rgby", ['I','II','III','IVA']):
    ax.scatter(dp[y['Stage'] == stage, 0], dp[y['Stage'] == stage,1], dp[y['Stage'] == stage, 2], c=c)

ax.scatter(dp[:,0], dp[:,1], dp[:,2])
plt.show()
