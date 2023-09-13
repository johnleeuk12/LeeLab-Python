# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 21:26:14 2023
Draw trajectories to compare
@author: Jong Hoon Lee
"""

import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
from scipy import stats
from sklearn.linear_model import TweedieRegressor, Ridge, ElasticNet, Lasso
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA, SparsePCA

from os.path import join as pjoin
from numba import jit, cuda


# %% load convdatas

Cb = np.load('Conv_btw.npy', allow_pickle= True).item()
C1 = np.load('Conv_R1.npy', allow_pickle= True).item()
C2 = np.load('Conv_R2_225.npy', allow_pickle= True).item()
C3 = np.load('Conv_R2_250.npy', allow_pickle= True).item()
C4 = np.load('Conv_R2_275.npy', allow_pickle= True).item()
C5 = np.load('Conv_R2_300.npy', allow_pickle= True).item()
C6 = np.load('Conv_R2_325.npy', allow_pickle= True).item()
# C6 = np.load('Conv_R2_325.npy', allow_pickle= True).item()
C7 = np.load('Conv_R2_350.npy', allow_pickle= True).item()




CR2 = np.load('Conv_R2.npy', allow_pickle= True).item()
Call = np.load('Conv_across.npy', allow_pickle= True).item()


# %% PCA

cmap3 = ['tab:purple','tab:blue','tab:red','tab:olive']

pca = {}
ax_sz = 4;
max_k = 20;
d_list = np.arange(600) > 179
d_list3 = np.arange(600) <= 179

d_list2 = d_list
fig, axs = plt.subplots(ax_sz,6,figsize = (20,20))

for f in np.arange(ax_sz):
    pca[f] = PCA(n_components=max_k)
    # R = C1[f][d_list2,:]+CR2[f][d_list2,:]
    
    R = CR2[f][d_list2,:]
    # R = R/2
    # R = Call[f][d_list2,:]
    test = pca[f].fit_transform(ndimage.gaussian_filter(R.T,[1,0]))        
    test = test.T
    for t in range(5):
        axs[f,t].plot(test[t,:],c = cmap3[f])
    axs[f,5].plot(np.cumsum(pca[f].explained_variance_ratio_))
    


# %% draw trajectories


from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Line3D
from matplotlib import cm
    
def draw_traj(traj,f,v,trmax,sc):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    # styles = ['solid', 'dotted', 'solid','dotted']
    # cmap_names = ['autumn','autumn','winter','winter']
    styles = ['solid','solid','dotted']
    cmap_names = ['autumn','winter','winter']
    for tr in np.arange(trmax):
        x = traj[f][tr][:,0]
        y = traj[f][tr][:,1]
        z = traj[f][tr][:,2]
        
        x = ndimage.gaussian_filter(x,1)
        y = ndimage.gaussian_filter(y,1)
        z = ndimage.gaussian_filter(z,1)            
            
        time = np.arange(len(x))
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
        
    
        
        # norm = plt.Normalize(time.min(), time.max())
        # cmap=plt.get_cmap(cmap_names[tr])
        # colors=[cmap(float(ii)/(n-1)) for ii in range(np.size(segments,0))]
        colors = cm.copper(np.linspace(0,1,trmax))
        
        # norm = BoundaryNorm([0,19,29,49,89,109],cmap.N)
        # lc = Line3DCollection(segments, cmap=cmap_names[tr], norm=norm,linestyle = styles[tr])
        lc = Line3DCollection(segments, color = colors[tr])#linestyle = styles[tr])
        # lc = Line3D(x,y,z, markevery = [0,19,29,49,89,109], color = colors[tr])#linestyle = styles[tr])
        if tr ==1:
            lc = Line3DCollection(segments, color = "red", linestyle = 'dotted')

        lc.set_array(time)
        lc.set_linewidth(2)
        ax.add_collection3d(lc)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        
        
        for m in [0]:
            ax.scatter(x[m], y[m], z[m], marker='o', color = "black")
        if tr == trmax:
            ax.auto_scale_xyz(x,y,z)
    if v ==1:
        for n in range(0, 100):
            if n >= 20 and n<50:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.elev = ax.elev+4.0 #pan down faster 
            if n >= 50 and n<80:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.elev = ax.elev+2.0 #pan down faster 
                ax.azim = ax.azim+4.0
            # if n >= 20 and n <= 22: 
            #     ax.set_xlabel('')
            #     ax.set_ylabel('') #don't show axis labels while we move around, it looks weird 
            #     ax.elev = ax.elev-2 #start by panning down slowly 
            # if n >= 23 and n <= 36: 
            #     ax.elev = ax.elev-1.0 #pan down faster 
            # if n >= 37 and n <= 60: 
            #     ax.elev = ax.elev-1.5 
            #     ax.azim = ax.azim+1.1 #pan down faster and start to rotate 
            # if n >= 61 and n <= 64: 
            #     ax.elev = ax.elev-1.0 
            #     ax.azim = ax.azim+1.1 #pan down slower and rotate same speed 
            # if n >= 65 and n <= 73: 
            #     ax.elev = ax.elev-0.5 
            #     ax.azim = ax.azim+1.1 #pan down slowly and rotate same speed 
            # if n >= 74 and n <= 76:
            #     ax.elev = ax.elev-0.2
            #     ax.azim = ax.azim+0.5 #end by panning/rotating slowly to stopping position
            if n >= 80: #add axis labels at the end, when the plot isn't moving around
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')
            # fig.suptitle(u'3-D Poincaré Plot, chaos vs random', fontsize=12, x=0.5, y=0.85)
            plt.savefig('Images/img' + str(n).zfill(3) + '.png',
                        bbox_inches='tight')


    
# %% projections onto PCA

    
traj = {};

sm = 10

for f in np.arange(1,4):
    if f == 0:
        R1 = ndimage.gaussian_filter(C1[2][d_list2,:].T,[4,0])
        R2 = ndimage.gaussian_filter(-C2[2][d_list2,:].T,[4,0])
        R3 = ndimage.gaussian_filter(-Cb[2][d_list2,:].T,[4,0])
    else:
        R1 = ndimage.gaussian_filter(C1[f][d_list2,:].T,[sm,0])
        R2 = ndimage.gaussian_filter(Cb[f][d_list2,:].T,[sm,0])
        R3 = ndimage.gaussian_filter(C2[f][d_list2,:].T,[sm,0])
        R4 = ndimage.gaussian_filter(C3[f][d_list2,:].T,[sm,0])
        R5 = ndimage.gaussian_filter(C4[f][d_list2,:].T,[sm,0])
        R6 = ndimage.gaussian_filter(C5[f][d_list2,:].T,[sm,0])
        R7 = ndimage.gaussian_filter(C6[f][d_list2,:].T,[sm,0])
        R8 = ndimage.gaussian_filter(C7[f][d_list2,:].T,[sm,0])




    # R1 = ndimage.gaussian_filter(D[0,5][d_list3,:].T,[2,0]) 
    # R2 = ndimage.gaussian_filter(D[1,5][d_list3,:].T,[2,0]) 
    traj[f] = {}
    traj[f][0] = np.dot(R1,pca[f].components_.T)  
    traj[f][1] = np.dot(R2,pca[f].components_.T)
    traj[f][2] = np.dot(R3,pca[f].components_.T)
    traj[f][3] = np.dot(R4,pca[f].components_.T)
    traj[f][4] = np.dot(R5,pca[f].components_.T)
    traj[f][5] = np.dot(R6,pca[f].components_.T)
    traj[f][6] = np.dot(R7,pca[f].components_.T)
    traj[f][7] = np.dot(R8,pca[f].components_.T)


for f in  np.arange(1,4):
    draw_traj(traj,f,0,8,0)


# %% Trajectory distance calculation


fig, axes = plt.subplots(ax_sz,1,figsize = (5,10))

xtime = np.arange(160)*5*1e-2 -2
colors = cm.copper(np.linspace(0,1,8))

distance = {};
for f in np.arange(1,ax_sz):
    for g in np.arange(7):
        distance[f,g] = np.linalg.norm(traj[f][g][:,0:3]-traj[f][6][:,0:3],axis = 1)
        if g == 0:
            axes[f].plot(xtime,distance[f,g], linestyle = 'dotted',color = colors[g])
        elif g == 1:
            axes[f].plot(xtime,distance[f,g], linestyle = 'solid',color = 'red')
        else: 
            axes[f].plot(xtime,distance[f,g], linestyle = 'solid',color = colors[g])




# %%

plt.close()
draw_traj(traj,3,1,8,0)
import IPython.display as IPdisplay
import glob
from PIL import Image as PIL_Image
import imageio

images = [PIL_Image.open(image) for image in glob.glob('images/*.png')]
file_path_name = 'images/GLM_kernel/R1vsR2_History.gif'
imageio.mimsave(file_path_name, images)


# %% PCA for all groups

cmap3 = ['tab:purple','tab:blue','tab:red','tab:olive']

pca = {}
ax_sz = 4;
max_k = 50;
d_list = np.arange(600) > 179
d_list3 = np.arange(600) <= 179

d_list2 = d_list
fig, axs = plt.subplots(ax_sz,6,figsize = (20,20))

C = {}
C[0] = C1
C[1] = Cb
C[2] = C2
C[3] = C3
C[4] = C4
C[5] = C5
C[6] = C6
C[7] = CR2

sm = 4
for f in np.arange(1,ax_sz):
    for g in np.arange(8):
        pca[g,f] = PCA(n_components=max_k)
        # R = C1[f][d_list2,:]+CR2[f][d_list2,:]
        
        # R = C[g][f][d_list2,:]
        R = ndimage.gaussian_filter(C[g][f][d_list2,:].T,[sm,0])

        # R = R/2
        # R = Call[f][d_list2,:]
        test = pca[g,f].fit_transform(ndimage.gaussian_filter(R,[1,0]))        
        test = test.T
for t in range(0,5):
    axs[f,t].plot(test[t,:],c = cmap3[f])
axs[f,5].plot(np.cumsum(pca[g,f].explained_variance_ratio_))

# %% subspace overlap, angle method
def list_shuffle(n,m,fract):
    p_list = {};
    p_list[0] = np.arange(n)
    p_list[1] = np.arange(n,m)
    
    for p in [0,1]:
        lp = int(np.floor(n*fract))
        shuffle  = np.random.choice([True, False],n, p = [lp/n, 1-lp/n])
        
        if p == 0:
            test = np.where(shuffle == False)
            for pp in test[0]:
                p_list[p][pp] = np.random.choice(p_list[1],1)
        elif p == 1:
            test = np.where(shuffle == False)
            for pp in test[0]:
                p_list[p][pp] = np.random.choice(p_list[0],1)
    
    return p_list


n_cv = 20   

trmax = 8
Overlap = {};
for f in np.arange(ax_sz):
    Overlap[f] = np.zeros((trmax,trmax,n_cv)); # PPC_IC
Overlap_across = np.zeros((trmax,trmax,n_cv));

O_mean = {}
O_std = {}
O_mean[0] = np.zeros((ax_sz,ax_sz));
O_std[0] = np.zeros((ax_sz,ax_sz));
O_mean[1] = np.zeros((ax_sz,ax_sz));
O_std[1] = np.zeros((ax_sz,ax_sz));


# n_list = {};
# n_list[0] = np.arange(95)
# n_list[1] = np.arange(95,len(good_list))

k1 = 0
k2 = 19

fig, axes = plt.subplots(1,ax_sz,figsize = (20,5))
for f in np.arange(1,ax_sz):
    for g1 in np.arange(trmax):
        for g2 in np.arange(trmax):
            S_value = np.zeros((1,20))
            for d in np.arange(0,5):
                S_value[0,d] = np.abs(np.dot(pca[g1,f].components_[d,:], pca[g2,f].components_[d,:].T))
                S_value[0,d] = S_value[0,d]/(np.linalg.norm(pca[g1,f].components_[d,:])*np.linalg.norm(pca[g2,f].components_[d,:]))
            
            Overlap[f][g1,g2,0] = np.max(S_value)
        
    imshowobj = axes[f-1].imshow(Overlap[f][:,:,0],cmap = "hot_r")
    imshowobj.set_clim(0.1, 0.5) #correct
plt.colorbar(imshowobj) #adjusts scale to value range, looks OK


# %% Calculate var explained percentage by PC 4-20 for R

# run R and PCA with separate subpopulations

fig, axes = plt.subplots(1,ax_sz,figsize = (20,5))

Overlap2 = {}
for f in np.arange(1,ax_sz):
    Overlap2[f] = np.zeros((trmax,trmax))
    for g1 in np.arange(trmax):        
        R0 = ndimage.gaussian_filter(C[g1][f][d_list2,:].T,[sm,0])
        # R0  = C[g1][f][d_list2,:].T
        V0  = 1-np.linalg.norm(R0 - np.dot(np.dot(R0,pca[g1,f].components_[0:6].T),
                                                                pca[g1,f].components_[0:6]))/np.linalg.norm(R0)
        for g2 in np.arange(trmax):
            Vcomp = 1-np.linalg.norm(R0 - np.dot(np.dot(R0,pca[g2,f].components_[0:6].T),
                                                                    pca[g2,f].components_[0:6]))/np.linalg.norm(R0)
            
            Overlap2[f][g1,g2] = Vcomp/V0
            
    imshowobj = axes[f-1].imshow(Overlap2[f][:,:],cmap = "hot_r")
    imshowobj.set_clim(0.1, 1) #correct
plt.colorbar(imshowobj) #adjusts scale to value range, looks OK















    
