# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:03:01 2023

@author: Jong Hoon Lee
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:08:42 2023

@author: Jong Hoon Lee
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:16:48 2023

        
       (trial,1) = stim onset time
       (trial,2) = rule (Rule 1 or 2)
       (trial,3) = contingency (1: go stim, 0: nogo stim)
       (trial,4) = Lick (1: lick, 0: No lick)
       (trial,5) = Correct choice (1: correct, 0: incorrect)
       (trial,6) = stage info (1: Task, 0: Conditioning)

@author: Jong Hoon Lee
"""

""" code updates:
    
    Sep 14 2022
    Allowing separation between conditions. This is to see if the same neurons with 
    action encoding in rule 1 shows in rule2 etc
    
    Sep 27 2022
    Major overhaul of code, including comments and separating analysis between
    across all trials and rule1 vs rule2

    
    Nov 09 2022
    C index determines how the code runs:
         0          :   Runs without trial history, across all rules
        -1          :   Runs with trial history, across all rules
        [1,2]       :   Separates rule 1 and rule 2
        [-3,-4]     :   Separates rule 1 and rule 2 but includes trial history
    
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
from numba import jit, cuda


# from matplotlib.colors import BoundaryNorm

# from sklearn.metrics import get_scorer_names, d2_tweedie_score, make_scorer


# %% File name and directory

# change fname for filename
fname = 'CaData_all_CS.mat'
fdir = 'D:\Python\Data'


# %% Helper functions for loading and selecting data
# 


np.seterr(divide = 'ignore') 

def load_matfile_Ca(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_CaData']
    return D_ppc 




def find_good_data_Ca(t_period):
    D_ppc = load_matfile_Ca()
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



# @jit(target_backend='cuda')                         


def import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind,ttr):    
    
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
    Yraw = {}
    Yraw = D_ppc[n,0]
    # Yraw2 = np.concatenate((np.flip(Yraw[0,0:3000],0),Yraw[0,:],Yraw[0,-3000:-1]),0)
    # sliding_w= np.lib.stride_tricks.sliding_window_view(np.arange(np.size(Yraw,1)+6000), 6000)
    # Ymed_wind = np.zeros((1,np.size(Yraw,1)))
    # for s in np.arange(np.size(Yraw,1)):
    #     Ymed_wind[0,s] = np.median(Yraw2[sliding_w[s,:]])
        
    # Yraw3 = Yraw-Ymed_wind+np.mean(Yraw)
    
    # Original Y calculation #####
    
    Y = np.zeros((N_trial,int(t_period/window)))
    for tr in range(N_trial):
        Y[tr,:] = Yraw[0,D_ppc[n,2][tr,0]-1 - int(prestim/window): D_ppc[n,2][tr,0] + int(t_period/window)-1 - int(prestim/window)]

    ####### analyzing Y including previous trial #####
    # Y = np.zeros((N_trial,int((2*(t_period+prestim))/window)))
    # Y[0,:] = np.concatenate((Yraw[0,D_ppc[n,2][0,0]-1 - int(prestim/window): D_ppc[n,2][0,0] + int(t_period/window)-1],
    #                         Yraw[0,D_ppc[n,2][0,0]-1 - int(prestim/window): D_ppc[n,2][0,0] + int(t_period/window)-1]))
    # for tr in range(1,N_trial):
    #     Y[tr,:] = np.concatenate((Yraw[0,D_ppc[n,2][tr-1,0]-1 - int(prestim/window): D_ppc[n,2][tr-1,0] + int(t_period/window)-1],
    #                               Yraw[0,D_ppc[n,2][tr,0]-1 - int(prestim/window): D_ppc[n,2][tr,0] + int(t_period/window)-1]))

    
    
    # for t in np.arange(int(t_period/window)):
    #     Y[:,t] = Y[:,t]- np/median(Y[:,t])


                
    # select analysis and model parameters with c_ind
    if ttr == 0 :
        c1 = 0
        c2 = 150
    elif ttr == 1: 
        c1  = 200
        c2 = 250
    elif ttr ==2:
        c1 =  np.size(X,0)-200
        c2 = c1+ 150
    else:
        c1 = D_ppc[n,4][0][0]-15
        c2 = D_ppc[n,4][0][0]+15

            
        
    Y = Y[c1:c2]
    X = X[c1:c2]
    L2 = L2[c1:c2]

    
    # Add reward  history
    Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
    Xpre = Xpre[:,None]
    Xpre2 = np.concatenate(([0,0],X[0:-2,2]*X[0:-2,1]),0)
    Xpre2 = Xpre2[:,None]
    # Add reward instead of action
    X2 = np.column_stack([X[:,0],X[:,3],
                          X[:,2]*X[:,1],Xpre]) 

    

    
    return X2,Y, L2, Rt


# %% Run main GLM code
"""     
Each column of X contains the following information:
    0 : contingency 
    1 : lick vs no lick
    2 : correct vs wrong
    3 : stim 1 vs stim 2
    4 : if exists, would be correct history (previous correct ) 

"""



t_period = 7000
prestim = 1000

window = 50 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 10 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [3]



D_ppc = load_matfile_Ca()
good_list = find_good_data_Ca(t_period)


# %% get neural data

lenx = 160 # Length of data, 8000ms, with a 50 ms window.
# good_list = np.arange(336)
D_all = np.zeros((len(good_list),lenx))
D = {}
trmax = 4

for tr in np.arange(trmax):
    D[0,tr] = np.zeros((len(good_list),lenx))
    D[1,tr] = np.zeros((len(good_list),lenx))


c_ind = 3
Y = {}
Ygo = {}
for tr in np.arange(trmax):
    print(tr)
    m = 0
    for n in good_list: 
        Ygo[m] = []
        n = int(n)
        X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind,tr)
        # D_all[m,:] = np.mean(Y,0)/(np.max(np.mean(Y,0)) + 0.5) # Soft normalisation, alpha = 0.5
        if tr == 3:
            Ygo[m] = Y[X[:,0] == 1,:]/(np.max(np.mean(Y,0)) + 0.5)
        D[0,tr][m,:] = np.mean(Y[X[:,0] == 0,:],0)/(np.max(np.mean(Y,0)) + 0.5)
        D[1,tr][m,:] = np.mean(Y[X[:,0] == 1,:],0)/(np.max(np.mean(Y,0)) + 0.5)
        m += 1

# %% Transform Ygo

Yraw = np.zeros((336,14,160))

for n in np.arange(336):
    Yraw[n,:,:] = Ygo[n][0:14,:]-np.median(Ygo[n][0:14,0:20])

for tr in np.arange(trmax):
    T = np.median(D[1,tr][:,0:20],1)
    D[1,tr] = D[1,tr]-T.reshape(336,1)


# %% PCA on individual groups

pca = {}
max_k = 25;
d_list = good_list > 118
d_list3 = good_list <= 118


d_list2 = d_list
# fig, axs = plt.subplots(trmax,6,figsize = (20,30))

sm = 4

for g in np.arange(trmax):
    pca[g] = PCA(n_components=max_k)
    R = ndimage.gaussian_filter(D[1,g][d_list2,:].T,[sm,0])
    test = pca[g].fit_transform(ndimage.gaussian_filter(R,[1,0]))        
    test = test.T
    # for t in range(0,5):
    #     axs[g,t].plot(test[t,:])
    # axs[g,5].plot(np.cumsum(pca[g].explained_variance_ratio_))


n_cv = 20   


Overlap = np.zeros((trmax,trmax,n_cv)); # PPC_IC

# O_mean = {}
# O_std = {}
# O_mean[0] = np.zeros((ax_sz,ax_sz));
# O_std[0] = np.zeros((ax_sz,ax_sz));
# O_mean[1] = np.zeros((ax_sz,ax_sz));
# O_std[1] = np.zeros((ax_sz,ax_sz));


# n_list = {};
# n_list[0] = np.arange(95)
# n_list[1] = np.arange(95,len(good_list))

k1 = 0
k2 = 19

fig, axes = plt.subplots(1,1,figsize = (10,10))
for g1 in np.arange(trmax):
   for g2 in np.arange(trmax):
       S_value = np.zeros((1,20))
       for d in np.arange(0,20):
           S_value[0,d] = np.abs(np.dot(pca[g1].components_[d,:], pca[g2].components_[d,:].T))
           S_value[0,d] = S_value[0,d]/(np.linalg.norm(pca[g1].components_[d,:])*np.linalg.norm(pca[g2].components_[d,:]))
            
       Overlap[g1,g2,0] = np.max(S_value)



imshowobj = axes.imshow(Overlap[:,:,0],cmap = "hot_r")
imshowobj.set_clim(0.1, 1) #correct
plt.colorbar(imshowobj) #adjusts scale to value range, looks OK
    
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
        if tr ==0:
            lc = Line3DCollection(segments, color = "red", linestyle = 'dotted')
        elif tr == 1:
            lc = Line3DCollection(segments, color = "black", linestyle = 'solid')

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



# %% trajectories

traj = {};

sm = 10
t1 = 0
t2 = 100
g = 0

R1 = ndimage.gaussian_filter(D[1,0][d_list2,t1:t2].T,[sm,0])
R2 = ndimage.gaussian_filter(D[1,1][d_list2,t1:t2].T,[sm,0])
traj[0] = {}
traj[0][0] = np.dot(R1,pca[g].components_.T)  
traj[0][1] = np.dot(R2,pca[g].components_.T)

for tr in np.arange(14):
    Rtr = ndimage.gaussian_filter(Yraw[d_list2,tr,t1:t2].T,[sm,0])
    traj[0][tr+2] = np.dot(Rtr,pca[g].components_.T)  




draw_traj(traj,0,0,16,0)













