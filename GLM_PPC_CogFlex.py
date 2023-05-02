# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:49:46 2023

1. Using Ca trial averaged population data for dimensionality reduction.
Projecting 4 trajectories onto the new low-dimensional space.
Indices:
    0 : No-Go
    1 : Go
    5 : Rule 1
    6 : Rule 2

Using ALL space PCA to project trajectories in different rules. 

@author: Jong Hoon Lee
"""


# import packages 

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

# from matplotlib.colors import BoundaryNorm

# from sklearn.metrics import get_scorer_names, d2_tweedie_score, make_scorer

# %% File name and directory

# change fname for filename

# fname = 'PPC_GLM_dataset_AllSession_FR_230209.mat'
fname = 'CaData_all_withlicktime.mat'
fdir = 'D:\Python\Data'
# fname = 'GLM_dataset_220824_new.mat'


# %% Helper functions for loading and selecting data
# 


np.seterr(divide = 'ignore') 
def load_matfile(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_dataset']
    return D_ppc 

def load_matfile_Ca(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_CaData']
    return D_ppc 


def find_good_data():
    D_ppc = load_matfile()
    good_list = []
    for n in range(np.size(D_ppc,0)):
        S_all = np.zeros((1,max(D_ppc[n,2][:,0])+t_period+100))
        for sp in np.array(D_ppc[n,0]):
            if sp < np.size(S_all,1):
                S_all[0,sp[0]-1] = 1  #spike time starts at 1 but indexing starts at 0
                
        if np.mean(S_all)*1e3>1:
            good_list = np.concatenate((good_list,[n]))
    return good_list


def find_good_data_Ca(t_period):
    D_ppc = load_matfile_Ca()
    # good_list = np.arange(np.size(D_ppc,0))
    good_list = []
    t_period = t_period+prestim

    for n in range(np.size(D_ppc,0)):
        N_trial = np.size(D_ppc[n,2],0)
    
    
    # re-formatting Ca traces
    
        Y = np.zeros((N_trial,int(t_period/window)))
        for tr in range(N_trial):
            Y[tr,:] = D_ppc[n,0][0,int(D_ppc[n,2][tr,0])-1 
                                 - int(prestim/window): int(D_ppc[n,2][tr,0])
                                 + int(t_period/window)-1 - int(prestim/window)]
        if np.mean(Y) > 0.5:
            good_list = np.concatenate((good_list,[n]))
    
    
    return good_list

def import_data_w_spikes(n,prestim,t_period,window,c_ind):
    D_ppc = load_matfile()
    S_all = np.zeros((1,max(D_ppc[n,2][:,0])+t_period+100))
    L_all = np.zeros((1,max(D_ppc[n,2][:,0])+t_period+100))
    N_trial = np.size(D_ppc[n,2],0)
    
    # extracting spikes from data
    for sp in np.array(D_ppc[n,0]):
        if sp < np.size(S_all,1):
            S_all[0,sp[0]-1] = 1  #spike time starts at 1 but indexing starts at 0
                
    
    S = np.zeros((N_trial,t_period))
    S_pre = np.zeros((N_trial,prestim))
    for tr in range(N_trial):
        S[tr,:] = S_all[0,D_ppc[n,2][tr,0]-1:D_ppc[n,2][tr,0]+t_period-1]
        S_pre[tr,:] = S_all[0,D_ppc[n,2][tr,0]-prestim-1:D_ppc[n,2][tr,0]-1]
    
    # extracting spikes, end
    
    # extracting licks, the same way
    for l in np.array(D_ppc[n,1]):
        if l < np.size(L_all,1):
            L_all[0,l[0]-1] = 1 
    
    L = np.zeros((N_trial,t_period+prestim))
    for tr in range(N_trial):
        L[tr,:] = L_all[0,D_ppc[n,2][tr,0]-prestim-1:D_ppc[n,2][tr,0]+t_period-1]
    
    
    X = D_ppc[n,2][:,2:6] # task variables
    Y = [];
    Y2 = [];
    S = np.concatenate((S_pre,S),1)
    t_period = t_period+prestim
    
    
    if c_ind !=3:
        if c_ind == 5:
            S = S[0:200,:]
            X = X[0:200,:]
            L = L[0:200,:]
        elif c_ind ==6:
            S = S[D_ppc[n,5][0][0]:,:]
            X = X[D_ppc[n,5][0][0]:,:]
            L = L[D_ppc[n,5][0][0]:,:]
        else:
    # remove conditioning trials     
            S = np.concatenate((S[0:200,:],S[D_ppc[n,5][0][0]:,:]),0)
            X = np.concatenate((X[0:200,:],X[D_ppc[n,5][0][0]:,:]),0)
            L = np.concatenate((L[0:200,:],L[D_ppc[n,5][0][0]:,:]),0)
    # only contain conditioning trials
    else:
        S = S[201:D_ppc[n,5][0][0]]
        X = X[201:D_ppc[n,5][0][0]]


    N_trial2 = np.size(S,0)

    # select analysis and model parameters with c_ind    
    
    if c_ind == -1:                
        # Adding previous trial correct vs wrong
        Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
        Xpre = Xpre[:,None]
        X = np.concatenate((X,Xpre),1)
        X2 = X
    elif c_ind ==-2:
        Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
        Xpre = Xpre[:,None]       
        # X2 = np.column_stack([X[:,0],(X[:,0]-1)*-1,
        #                      X[:,3],(X[:,3]-1)*-1,
        #                      X[:,2]*X[:,1],Xpre])
        X2 = np.column_stack([X[:,0],X[:,3],
                             X[:,2]*X[:,1],Xpre])  
    
    
    
    L2 = []
    for w in range(int(t_period/window)):
        l = np.sum(L[:,range(window*w,window*(w+1))],1)
        L2 = np.concatenate((L2,l))
        y = np.mean(S[:,range(window*w,window*(w+1))],1)*1e3
        y2 = np.sum(S[:,range(window*w,window*(w+1))],1)
        Y = np.concatenate((Y,y))
        Y2 = np.concatenate((Y2,y2))
        
    Y = np.reshape(Y,(int(t_period/window),N_trial2)).T
    Y2 = np.reshape(Y2,(int(t_period/window),N_trial2)).T
    L2 = np.reshape(L2,(int(t_period/window),N_trial2)).T
    return X2, Y, Y2, L2

def import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind):
    # D_ppc = load_matfile_Ca()
    
    
    L_all = np.zeros((1,int(np.floor(D_ppc[n,3][0,max(D_ppc[n,2][:,0])]*1e3))+t_period+100))
    N_trial = np.size(D_ppc[n,2],0)
    
    # extracting licks, the same way
    for l in np.floor(D_ppc[n,1]*1e3):
        l = int(l) 
        if l < np.size(L_all,1):
            L_all[0,l-1] = 1 
    
    L = np.zeros((N_trial,t_period+prestim))
    for tr in range(N_trial):
        stim_onset = int(np.round(D_ppc[n,3][0,D_ppc[n,2][tr,0]]*1e3))
        lick_onset = int(np.round(D_ppc[n,3][0,D_ppc[n,2][tr,3]]*1e3))
        lick_onset = lick_onset-stim_onset
        L[tr,:] = L_all[0,stim_onset-prestim-1:stim_onset+t_period-1]
        
        # reformatting lick rates
    L2 = []
    for w in range(int((t_period+prestim)/window)):
        l = np.sum(L[:,range(window*w,window*(w+1))],1)
        L2 = np.concatenate((L2,l)) 
            
    L2 = np.reshape(L2,(int((t_period+prestim)/window),N_trial)).T


    X = D_ppc[n,2][:,2:6] # task variables
    Rt =  D_ppc[n,5] # reward time relative to stim onset, in seconds
    t_period = t_period+prestim
    
    # re-formatting Ca traces
    
    Y = np.zeros((N_trial,int(t_period/window)))
    for tr in range(N_trial):
        Y[tr,:] = D_ppc[n,0][0,D_ppc[n,2][tr,0]-1 - int(prestim/window): D_ppc[n,2][tr,0] + int(t_period/window)-1 - int(prestim/window)]
    


                
    # select analysis and model parameters with c_ind
    
    if c_ind !=3:
        if c_ind == 5:
            Y = Y[0:200,:]
            X = X[0:200,:]
            L2 = L2[0:200,:]
        elif c_ind ==6:
            Y = Y[D_ppc[n,4][0][0]:,:]
            X = X[D_ppc[n,4][0][0]:,:]
            L2 = L2[D_ppc[n,4][0][0]:,:]
        else:
    # remove conditioning trials     
            Y = np.concatenate((Y[0:200,:],Y[D_ppc[n,4][0][0]:,:]),0)
            X = np.concatenate((X[0:200,:],X[D_ppc[n,4][0][0]:,:]),0)
            L2 = np.concatenate((L2[0:200,:],L2[D_ppc[n,4][0][0]:,:]),0)          
    else:
    # only contain conditioning trials    
        Y = Y[201:D_ppc[n,4][0][0]]
        X = X[201:D_ppc[n,4][0][0]]
        L2 = L2[201:D_ppc[n,4][0][0]]

    
    # Add reward  history
    Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
    Xpre = Xpre[:,None]
    Xpre2 = np.concatenate(([0,0],X[0:-2,2]*X[0:-2,1]),0)
    Xpre2 = Xpre2[:,None]
    # Add reward instead of action
    X2 = np.column_stack([X[:,0],X[:,3],
                          X[:,2]*X[:,1],Xpre]) 

    

    
    return X2,Y, L2, Rt

# %% 
t_period = 7000
prestim = 1000

window = 50 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 10 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [-2]



if ca ==0:
    D_ppc = load_matfile()
    good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    good_list = find_good_data_Ca(t_period)
    
    
    
lenx = 160 # Length of data, 8000ms, with a 50 ms window.
D_all = np.zeros((len(good_list),lenx))
D = {}
D[0,5] = np.zeros((len(good_list),lenx))
D[1,5] = np.zeros((len(good_list),lenx))
D[0,6] = np.zeros((len(good_list),lenx))
D[1,6] = np.zeros((len(good_list),lenx))
m = 0;    
for n in good_list:
    n = int(n)
    X,Y,L,Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,-2)
    D_all[m,:] = np.mean(Y,0)/(np.max(np.mean(Y,0)) + 0.5) # Soft normalisation, alpha = 0.5
    for c_ind in [5,6]:
        X,Y,L,Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
        D[0,c_ind][m,:] = np.mean(Y[X[:,0] == 0,:],0)/(np.max(np.mean(Y[X[:,0] == 0,:],0)) + 0.5)
        D[1,c_ind][m,:] = np.mean(Y[X[:,0] == 1,:],0)/(np.max(np.mean(Y[X[:,0] == 0,:],0)) + 0.5)
    m += 1
    


# %% Run PCA, can separate PPC_AC (d_list2) and PPC_IC (d_list1)
f = 0

d_list1 = good_list < 179
d_list2 = good_list > 179
pca = {};
pca[f] = PCA(n_components=80) 
# test = pca[f].fit_transform(ndimage.gaussian_filter(Convdata[f][:,:].T,[2,0])) # change to [2,0] if SU data, else, [1,0]
test = pca[f].fit_transform(ndimage.gaussian_filter(D_all[d_list2,:].T,[1,0]))


    
test = test.T
fig, axs = plt.subplots(1,6,figsize = (20,5))
for t in range(5):
    axs[t].plot(test[t,:],c = 'tab:purple')
axs[5].plot(np.cumsum(pca[f].explained_variance_ratio_[0:5]))


R = ndimage.gaussian_filter(D_all.T,[1,0])


traj = {}
traj[f] = {}
# traj[f][0] = pca[f].fit_transform(R)
traj[f][0] = np.dot(ndimage.gaussian_filter(D[0,5][d_list2,:].T,[1,0]),pca[f].components_.T)  
traj[f][1] = np.dot(ndimage.gaussian_filter(D[1,5][d_list2,:].T,[1,0]),pca[f].components_.T)  
traj[f][2] = np.dot(ndimage.gaussian_filter(D[0,6][d_list2,:].T,[1,0]),pca[f].components_.T)  
traj[f][3] = np.dot(ndimage.gaussian_filter(D[1,6][d_list2,:].T,[1,0]),pca[f].components_.T)  
        
    
    

# %% draw trajectories


from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

    
def draw_traj(traj,f,v):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    styles = ['solid', 'dotted', 'solid','dotted']
    cmap_names = ['autumn','autumn','winter','winter']
    for tr in [0,1,2,3]:
        x = traj[f][tr][:,0]
        y = traj[f][tr][:,1]
        z = traj[f][tr][:,2]
        if ca == 0:
            x = ndimage.gaussian_filter(x,1)
            y = ndimage.gaussian_filter(y,1)
            z = ndimage.gaussian_filter(z,1)            
            
        time = np.arange(len(x))
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
        
    
        
        # norm = plt.Normalize(time.min(), time.max())
        cmap=plt.get_cmap(cmap_names[tr])
        # colors=[cmap(float(ii)/(n-1)) for ii in range(np.size(segments,0))]
        
        
        norm = BoundaryNorm([0,19,29,49,89,109],cmap.N)
        lc = Line3DCollection(segments, cmap=cmap_names[tr], norm=norm,linestyle = styles[tr])
        lc.set_array(time)
        lc.set_linewidth(2)
        ax.add_collection3d(lc)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.auto_scale_xyz(x,y,z)


# %% 
draw_traj(traj,0,0)
    
    

    
    
    
    
    
    
    
    
    
    
