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

from rastermap import Rastermap, utils
from scipy.stats import zscore
# from matplotlib.colors import BoundaryNorm

# from sklearn.metrics import get_scorer_names, d2_tweedie_score, make_scorer


# %% File name and directory

# change fname for filename
# fname = 'CaData_all_withlicktime_correctedv2.mat'


# fname = 'CaData_all_all_session_v2_corrected.mat'
fname = 'CaData_all_session_v3_corrected.mat'

# fname = 'CaData_PIC_2.mat'


fdir = 'D:\Python\Data'


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



@jit(target_backend='cuda')                         
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
    # for l in np.array(D_ppc[n,1]):
    #     if l < np.size(L_all,1):
    #         L_all[0,l[0]-1] = 1 
    
    L = np.zeros((N_trial,t_period+prestim))
    for tr in range(N_trial):
        L[tr,:] = L_all[0,D_ppc[n,2][tr,0]-prestim-1:D_ppc[n,2][tr,0]+t_period-1]
    
    
    X = D_ppc[n,2][:,2:6] # task variables
    Y = [];
    Y2 = [];
    S = np.concatenate((S_pre,S),1)
    t_period = t_period+prestim
    
    
    if c_ind !=3:
    # remove conditioning trials     
        S = np.concatenate((S[0:200,:],S[D_ppc[n,5][0][0]:,:]),0)
        X = np.concatenate((X[0:200,:],X[D_ppc[n,5][0][0]:,:]),0)
        L = np.concatenate((L[0:200,:],L[D_ppc[n,5][0][0]:,:]),0)

    # only contain conditioningt trials
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

def import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind,ttr):    
    
    L_all = np.zeros((1,int(np.floor(D_ppc[n,3][0,max(D_ppc[n,2][:,0])]*1e3))+t_period+100))
    N_trial = np.size(D_ppc[n,2],0)
    
    # extracting licks, the same way
    # for l in np.floor(D_ppc[n,1]*1e3):
    #     # l = int(l) 
    #     if int(l) < np.size(L_all,1):
    #         L_all[0,l-1] = 1 
    
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
    
    if c_ind ==0:             
    # remove conditioning trials 
        Y = np.concatenate((Y[0:200,:],Y[D_ppc[n,4][0][0]:,:]),0)
        X = np.concatenate((X[0:200,:],X[D_ppc[n,4][0][0]:,:]),0)
        L2 = np.concatenate((L2[0:200,:],L2[D_ppc[n,4][0][0]:,:]),0)
    elif c_ind == 1:        
        Y = Y[0:200,:]
        X = X[0:200,:]
        L2 = L2[0:200,:]
    elif c_ind ==2:
        Y = Y[D_ppc[n,4][0][0]:,:]
        X = X[D_ppc[n,4][0][0]:,:]
        L2 = L2[D_ppc[n,4][0][0]:,:]
    elif c_ind ==3:
    # only contain conditioning trials    
        # Y = Y[201:D_ppc[n,4][0][0]]
        # X = X[201:D_ppc[n,4][0][0]]
        # L2 = L2[201:D_ppc[n,4][0][0]]
        # Y = Y[201:250]
        # X = X[201:250]
        # L2 = L2[201:250]
        # if ttr == 0:
        #     c1 = 0
        #     c2 = 150
        # elif ttr == 1:
        #     c1 = 200
        #     c2 = 250
        # else:
        #     c1 = 200+(ttr-2)*25
        #     c2 = c1 +50
            
        # if ttr < 5:            
        #     c1 = ttr*50
        #     c2 = c1 +50
        # else:
        #     if np.size(X,0)-400 >50:                
        #         c1  = np.size(X,0)-200+(ttr-5)*50
        #     else:
        #         c1 = 250 + (ttr-5)*50
        #     if ttr == 8:
        #         c2 = np.size(X,0)
        #     else:
        #         c2 = c1+50
        #     c1 = c1+25
        #     c2 = c2+25
        if ttr == -1:
            c1 = 200 
            c2 = D_ppc[n,4][0][0] +10
            # c1 = D_ppc[n,4][0][0]-20
            # c2 = D_ppc[n,4][0][0] +20
            
        elif ttr < 4:            
            c1 = ttr*50
            c2 = c1 +50
        else:
            c1 = D_ppc[n,4][0][0]+(ttr-4)*50
            # c1  = np.size(X,0)-200+(ttr-4)*50
            c2 = c1+ 50
            if ttr == 7:
                c2 = np.size(X,0)
            else:
                c2 = c1+50
            # c1 = c1+25
            # c2 = c2+25
            
            
            
        
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



t_period = 6000
prestim = 2000

window = 50 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 10 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [3]



if ca ==0:
    D_ppc = load_matfile()
    # good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    good_list = find_good_data_Ca(t_period)
    
    
# %% get neural data

lenx = 160 # Length of data, 8000ms, with a 50 ms window.
# good_list = np.arange(726)
# good_list = np.arange(np.size(D_ppc,0))

D_all = np.zeros((len(good_list),lenx))
D = {}
trmax = 8
alpha = 0.2
for tr in np.arange(trmax):
    D[0,tr] = np.zeros((len(good_list),lenx))
    D[1,tr] = np.zeros((len(good_list),lenx))
    D[2,tr] = np.zeros((len(good_list),lenx))



for ind in [0,1,2]:
    D[ind,trmax] = np.zeros((len(good_list),lenx))
    D[ind,trmax+1] = np.zeros((len(good_list),lenx))

c_ind = 3
Y = {}
for tr in np.arange(trmax):
    print(tr)
    m = 0
    ttr = tr
    if tr == 4:
        ttr = -1
    elif tr >4:
        ttr = tr-1
        
    for n in good_list: 
        n = int(n)
        X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind,ttr)
        # D[0,tr][m,:] = np.mean(Y[X[:,0] == 0,:],0)/(np.max(np.mean(Y,0)) + alpha)
        # D[1,tr][m,:] = np.mean(Y[X[:,0] == 1,:],0)/(np.max(np.mean(Y,0)) + alpha)
        # D[2,tr][m,:] = np.mean(Y,0)/(np.max(np.mean(Y,0)) + alpha)
        for ind in [0,1]:
            # D[ind,tr][m,:] = (np.mean(Y[X[:,0] == ind,:],0)-np.mean(Y[X[:,0] == ind,10:30]))
            D[ind,tr][m,:] = np.mean(Y[X[:,0] == ind,:],0)
            D[ind,tr][m,:] = D[ind,tr][m,:]/((np.max(np.mean(Y,0))) + alpha) # for original trajectories with Go+ NG
            # D[ind,tr][m,:] = D[ind,tr][m,:]/((np.max(np.mean(Y[X[:,0] == 1,:],0))) + alpha)

            # D[ind,tr][m,:] = D[ind,tr][m,:]/(np.max(D[ind,tr][m,:]) + alpha)

        # D[2,tr][m,:] = np.mean(Y,0)-np.mean(Y[:,10:30])
        # D[2,tr][m,:] = D[2,tr][m,:]/(np.max(np.mean(Y,0)) + alpha)
        m += 1
        
for c_ind in[1,2]:
    m = 0
    for n in good_list: 
        n = int(n)
        X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind,ttr)
        for ind in [0,1]:
            # D[ind,trmax+c_ind-1][m,:] = np.mean(Y[X[:,0] == ind,:],0)-np.mean(Y[X[:,0] == ind,10:30])
            D[ind,trmax+c_ind-1][m,:] = np.mean(Y[X[:,0] == ind,:],0)

            D[ind,trmax+c_ind-1][m,:] = D[ind,trmax+c_ind-1][m,:]/(np.max(np.mean(Y,0)) + alpha)
            # D[ind,trmax+c_ind-1][m,:] = D[ind,trmax+c_ind-1][m,:]/((np.max(Y)-np.min(Y)) + alpha)
        # 
        # D[2,trmax+c_ind-1][m,:] = (np.mean(Y,0)-np.mean(Y[:,10:30]))/(np.max(np.mean(Y,0)) + alpha)

        m += 1
        

c_ind =3


# %% draw Go and No-Go

# fig, axes = plt.subplots(1,1, figsize = (10,10))
# axes.plot(Y3)
d_list = good_list > 195

d_list3 = good_list <= 179
fig, axes = plt.subplots(1,1, figsize = (10,10))
for tr in [8,4,9]:
    axes.plot(np.mean(D[1,tr][d_list3,:],0))

# %% Plot Go vs No-Go
d_list = good_list > 179
d_list3 = good_list <= 179

d_list2 = d_list
tr_ind = 0
fig, axes = plt.subplots(figsize =(10,10))

cmap = ["tab:blue","tab:red"]
xaxis = np.linspace(-2,6,lenx+1)
xaxis  = xaxis[1:]
for c in [0,1]:
    gD = ndimage.gaussian_filter(D[c,tr_ind][d_list2,:],[0,3])
    sD = np.std(gD,0)/np.sqrt(np.size(gD,0))
    axes.plot(xaxis,np.mean(gD,0),color = cmap[c],linewidth = 4)
    axes.fill_between(xaxis,np.mean(gD,0)+sD,np.mean(gD,0)-sD,color = cmap[c], alpha = 0.3)
    # axes.set_ylim([-0.10, 1.05])
    # axes.set_ylim([-0.15, 0.40])


# %% plot example neurons
d_list = good_list > 179

d_list3 = good_list <= 179


d_list2 = d_list
D_r = {}
sm = 1
for g in [0,1]:
    D_r[g] = D[1,trmax+g][d_list2,:]
    D_r[g] = ndimage.gaussian_filter(D_r[g],[sm,0])

max_ind = np.argmax(np.abs(D_r[0][:,40:]),1)
peaks = np.zeros((np.size(D_r[0],0),1))
max_ind = max_ind +40
for n in np.arange(np.size(D_r[0],0)):
    peaks[n,0] = D_r[0][n,max_ind[n]]

max_peaks = np.argsort(peaks[:,0])

fig, axes = plt.subplots(1,1,figsize = (10,10))
clim = [0.2,0.8]
im1 = axes.imshow(D_r[0][max_peaks,:], clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im1, cax=cbar_ax)

# %% 

d_list = good_list > 179

d_list3 = good_list <= 179


d_list2 = d_list3
D_r = {}
sm = 2
g_ind = 1
# for g in [0,1]:
#     # D_r[g] = zscore(D[g_ind,trmax+g][d_list2,:],axis =1)
#     D_r[g] = D[g_ind,trmax+g][d_list2,:]
#     D_r[g] = ndimage.gaussian_filter(D_r[g],[sm,0])
#     # D_r[g] = zscore(D_r[g],axis =1)
# # D_r[2] = zscore(D[g_ind,4][d_list2,:], axis = 1)
# D_r[2] = D[g_ind,4][d_list2,:]

trmax = 8


for g in np.arange(trmax+2):
    # D_r[g] = zscore(D[g_ind,trmax+g][d_list2,:],axis =1)
    D_r[g] = D[g_ind,g][d_list2,:]
    D_r[g] = ndimage.gaussian_filter(D_r[g],[sm,0])
    # D_r[g] = zscore(D_r[g],axis =1)
# D_r[2] = zscore(D[g_ind,4][d_list2,:], axis = 1)
# D_r[2] = D[g_ind,4][d_list2,:]



max_ind = np.argmax(D_r[0][:,:],1)
max_peaks = np.argsort(max_ind)


# fig, axes = plt.subplots(1,3,figsize = (20,10))
clim = [-1,1.5]

# im1 = axes[0].imshow(D_r[0][max_peaks,:],clim = clim, aspect = "auto", interpolation = "None",cmap = "gray_r")
# im1 = axes[1].imshow(D_r[2][max_peaks,:],clim = clim, aspect = "auto", interpolation = "None",cmap = "gray_r")
# im1 = axes[2].imshow(D_r[1][max_peaks,:],clim = clim, aspect = "auto", interpolation = "None",cmap = "gray_r")

# # im1 = axes[0].imshow(zscore(D_r[0][max_peaks,:],axis = 1),clim = clim, aspect = "auto", interpolation = "None",cmap = "gray_r")
# # im1 = axes[1].imshow(zscore(D_r[2][max_peaks,:],axis = 1),clim = clim, aspect = "auto", interpolation = "None",cmap = "gray_r")
# # im1 = axes[2].imshow(zscore(D_r[1][max_peaks,:],axis = 1),clim = clim, aspect = "auto", interpolation = "None",cmap = "gray_r")
# fig.subplots_adjust(right=0.85)
# cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
# fig.colorbar(im1, cax=cbar_ax)



fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(30, 20))
for ind, ax in zip(np.arange(8), axs.ravel()):
    ax.imshow(zscore(D_r[ind][max_peaks, :],axis = 1),clim = clim, cmap="viridis", vmin=clim[0], vmax=clim[1], aspect="auto")


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 20))
im1 =axs[0].imshow(zscore(D_r[8][max_peaks, :],axis = 1),clim = clim, cmap="viridis",  vmin=clim[0], vmax=clim[1], aspect="auto")
im1 =axs[1].imshow(zscore(D_r[9][max_peaks, :],axis = 1),clim = clim, cmap="viridis",  vmin=clim[0], vmax=clim[1], aspect="auto")

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im1, cax=cbar_ax)

# %% rastermap



# spks is neurons by time
# spks = np.load("spks.npy").astype("float32")
# spks = zscore(spks, axis=1)

# spks = zscore(D_r[0],axis = 1)
spks = D_r[8]
# fit rastermap
# note that D_r is already normalized
model = Rastermap(n_PCs=64,
                  locality=0.75,
                  time_lag_window=5,
                  n_clusters = 50,
                  grid_upsample=5, keep_norm_X = False).fit(spks)
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
X_embedding = zscore(utils.bin1d(spks[isort], bin_size=25, axis=0), axis=1)

# plot

# plot embedding 


clim = [0,1.5]
# clim = [0.1,0.9]

# fig, axes = plt.subplots(1,3,figsize = (20,10))
# clim = [0.3,1.0]
# im1 = axes[0].imshow(zscore(D_r[0][isort, :],axis = 1),clim = clim, cmap="gray_r", vmin=0, vmax=1.2, aspect="auto")
# axes[2].imshow(zscore(D_r[4][isort, :],axis = 1),clim = clim, cmap="gray_r", vmin=0, vmax=1.2, aspect="auto")
# axes[1].imshow(zscore(D_r[6][isort, :],axis = 1),clim = clim, cmap="gray_r", vmin=0, vmax=1.2, aspect="auto")

fig, axs = plt.subplots(nrows=1, ncols=8, figsize=(90, 10))
for ind, ax in zip(np.arange(8), axs.ravel()):
    ax.imshow(zscore(D_r[ind][isort, :],axis = 1),clim = clim, cmap="gray_r",  vmin=clim[0], vmax=clim[1], aspect="auto")

    
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 20))
axs[0].imshow(zscore(D_r[8][isort, :],axis = 1),clim = clim, cmap="viridis",  vmin=clim[0], vmax=clim[1], aspect="auto")
axs[1].imshow(zscore(D_r[9][isort, :],axis = 1),clim = clim, cmap="viridis",  vmin=clim[0], vmax=clim[1], aspect="auto")
# axs[0].imshow(D_r[8][isort, :],clim = clim, cmap="viridis",  vmin=clim[0], vmax=clim[1], aspect="auto")
# axs[1].imshow(D_r[9][isort, :],clim = clim, cmap="viridis",  vmin=clim[0], vmax=clim[1], aspect="auto")



v = clim

# im1 = axes[0].imshow(D_r[0][isort, :],clim = clim, cmap="gray_r", vmin=v[0], vmax=v[1], aspect="auto")
# axes[2].imshow(D_r[1][isort, :],clim = clim, cmap="gray_r", vmin=v[0], vmax=v[1], aspect="auto")
# axes[1].imshow(D_r[2][isort, :],clim = clim, cmap="gray_r", vmin=v[0], vmax=v[1], aspect="auto")

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im1, cax=cbar_ax)


# D_hat = np.dot(model.Usv,model.Vsv.T)

# fig, axs = plt.subplots(figsize = (10,10))
# axs.plot(np.mean(D_hat,0))
# embed = {};
# for ind in np.arange(8):
#     embed[ind] = np.dot(D_r[ind],model.Vsv)


# raster_model = model


# fig = plt.figure(figsize=(12,5))
# ax = fig.add_subplot(111)
# ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")

# embe_cluster = model.embedding_clust
# ax.imshow(spks[isort, :], cmap="gray_r", vmin=0, vmax=1.2, aspect="auto")


# %% PCA on individual groups

pca = {}
max_k = 20;
# d_list = np.logical_and(good_list > 179, good_list < 600)
d_list = good_list > 195

d_list3 = good_list <= 195

trmax = 8

d_list2 = d_list
# d_list2 = good_list>-1
fig, axs = plt.subplots(trmax+2,6,figsize = (20,30))

sm = 0
R = {}
for g in  np.arange(trmax+2):
    pca[g] = PCA(n_components=20)
    # R[g] = np.concatenate((ndimage.gaussian_filter(D[1,g][d_list2,:].T,[sm,0]),ndimage.gaussian_filter(D[0,g][d_list2,:].T,[sm,0])),1)
    # R[g] = ndimage.gaussian_filter(D[1,g][d_list2,:].T,[sm,0]) + ndimage.gaussian_filter(D[1,g][d_list2,:].T,[sm,0])
    R[g] = D[1,g][d_list2,:].T +D[1,g][d_list2,:].T
    # R[g] = R[g]/2
    # R[g] = np.concatenate((D[0,g][d_list2,:].T,D[1,g][d_list2,:].T),1)
# 
    # R[g] =ndimage.gaussian_filter(D[1,g][d_list2,:].T,[sm,0])
    # R = ndimage.gaussian_filter(D_r[g][:,:].T,[sm,0])
    # R = ndimage.gaussian_filter(zscore(D[1,g][d_list2,:], axis = 1),[sm,0])
    test = pca[g].fit_transform(ndimage.gaussian_filter(R[g],[1,0]))        
    test = test.T
    for t in range(0,5):
        axs[g,t].plot(test[t,:], linewidth = 4)
    axs[g,5].plot(np.cumsum(pca[g].explained_variance_ratio_), linewidth = 4)
    
# g = 2
# fig, axs = plt.subplots(1,1, figsize = (10,10))
# axs.plot(np.mean(ndimage.gaussian_filter(D[1,g][d_list2,:].T,[sm,0]),1))
# axs.plot(np.mean(ndimage.gaussian_filter(D[0,g][d_list2,:].T,[sm,0]),1))
# axs.plot(np.mean(ndimage.gaussian_filter(R[g],[sm,0]),1))
# %% creating concatenated R



d_list = good_list > 195
d_list3 = good_list <= 195

d_list2 = d_list
sm = 0
# R =  np.empty( shape=(160,0) )
R =  np.empty( shape=(0,np.sum(d_list2)) )
label = np.empty( shape=(2,0) )
tr_max = 8
for tr in np.arange(tr_max):
    if tr != 3:
        for ind in [0,1]:
            R = np.concatenate((R,ndimage.gaussian_filter(D[ind,tr][d_list2,:].T,[sm,0])),0)
            lbl = np.concatenate((np.ones((1,np.sum(d_list2)))*tr,np.ones((1,np.sum(d_list2)))*ind),0)
            label = np.concatenate((label,lbl),1)


# fig, axs = plt.subplots(1,6,figsize = (20,10))

# RT = np.dot(R,raster_model.Usv)

pca_all = PCA(n_components=64)
test = pca_all.fit_transform(R)  

W =pca_all.components_


# %% subspace overlap 
from scipy import linalg

n_cv = 20   
trmax = 8


Overlap = np.zeros((trmax-1,trmax-1,n_cv)); # PPC_IC
Overlap_across = np.zeros((trmax,trmax,n_cv));
 
# O_mean = {}
# O_std = {}
# O_mean[0] = np.zeros((ax_sz,ax_sz));
# O_std[0] = np.zeros((ax_sz,ax_sz));
# O_mean[1] = np.zeros((ax_sz,ax_sz));
# O_std[1] = np.zeros((ax_sz,ax_sz));


# n_list = {};
# n_list[0] = np.arange(95)
# n_list[1] = np.arange(95,len(good_list))

k1 = 10
k2 = 19

U = {}
for g in  np.arange(trmax+2):
    U[g], s, Vh = linalg.svd(R[g].T)

fig, axes = plt.subplots(1,1,figsize = (10,10))
for g1 in [0,1,2,4,5,6,7]: #np.arange(trmax):
   for g2 in [0,1,2,4,5,6,7]: # np.arange(trmax):
       S_value = np.zeros((1,k1))
       for d in np.arange(0,k1):
           S_value[0,d] = np.abs(np.dot(pca[g1].components_[d,:], pca[g2].components_[d,:].T))
           S_value[0,d] = S_value[0,d]/(np.linalg.norm(pca[g1].components_[d,:])*np.linalg.norm(pca[g2].components_[d,:]))
           # S_value[0,d] = np.abs(np.dot( U[g1][:,d], U[g2][:,d]))
           # S_value[0,d] = S_value[0,d]/(np.linalg.norm(U[g1][:,d].T)*np.linalg.norm(U[g1][:,d].T))
       if g1 < 4:
           if g2 < 4:
               Overlap[g1,g2,0] = np.max(S_value)
           elif g2 >= 4:
               Overlap[g1,g2-1,0] = np.max(S_value)
       elif g1 >= 4:
           if g2 < 4:
               Overlap[g1-1,g2,0] = np.max(S_value)
           elif g2 >= 4:
               Overlap[g1-1,g2-1,0] = np.max(S_value)
           
        
imshowobj = axes.imshow(Overlap[:,:,0],cmap = "hot_r")
imshowobj.set_clim(0.2, 0.9) #correct
plt.colorbar(imshowobj) #adjusts scale to value range, looks OK

# plt.savefig("Fraction of neurons "+ str(ind) + "tv" + str(p) + ".svg")
# import os

# dirname = "D:\Python\Figures\New_2"
imgname = "PPCIC_subspace"
plt.savefig(imgname + ".svg")

    
# %% subsplace overlap 2nd method


# g1 = 2
# g2 = 5

# V1 = 1-np.linalg.norm(R[g1] - np.dot(np.dot(R[g1],pca[g2].components_.T),
#                                                                 pca[g2].components_))/np.linalg.norm(R[g1])
        
# V2 = 1-np.linalg.norm(R[g1] - np.dot(np.dot(R[g1],pca[g1].components_.T),
#                                                                 pca[g1].components_))/np.linalg.norm(R[g1])
Overlap = np.zeros((trmax-1,trmax-1,n_cv)); # PPC_IC
k1 = 0
k2 = 10
fig, axes = plt.subplots(1,1,figsize = (10,10))

U = {}
for g in  np.arange(trmax+2):
    U[g], s, Vh = linalg.svd(R[g].T)



for g1 in [0,1,2,4,5,6,7]: #np.arange(trmax):
   for g2 in [0,1,2,4,5,6,7]: # np.arange(trmax):
        # V1 = 1-np.linalg.norm(R[g1] - np.dot(np.dot(R[g1],pca[g2].components_.T),pca[g2].components_))/np.linalg.norm(R[g1])
        V1 = 1-np.linalg.norm(R[g1] - np.dot(np.dot(R[g1],U[g2][:,k1:k2]),U[g2][:,k1:k2].T))/np.linalg.norm(R[g1])
        # V2 = 1-np.linalg.norm(R[g1] - np.dot(np.dot(R[g1],pca[g1].components_.T),pca[g1].components_))/np.linalg.norm(R[g1])
        V2 = 1-np.linalg.norm(R[g1] - np.dot(np.dot(R[g1],U[g1][:,k1:k2]),U[g1][:,k1:k2].T))/np.linalg.norm(R[g1])
        if g1 < 4:
            if g2 < 4:
                Overlap[g1,g2,0] = V1/V2
            elif g2 >= 4:
                Overlap[g1,g2-1,0] = V1/V2
        elif g1 >= 4:
            if g2 < 4:
                Overlap[g1-1,g2,0] = V1/V2
            elif g2 >= 4:
                Overlap[g1-1,g2-1,0] = V1/V2
        
imshowobj = axes.imshow(Overlap[:,:,0],cmap = "hot_r")
imshowobj.set_clim(0.5, 0.8) #correct
plt.colorbar(imshowobj) #adjusts scale to value range, looks OK

test =np.dot(np.dot(R[g1],U[g1][:,:k]),U[g1][:,:k].T)

fig ,axes = plt.subplots(1,1, figsize = (10,10)) 
axes.plot(np.mean(test,1))   
axes.plot(np.mean(R[g1],1))   


# U, s, Vh = linalg.svd(R[g1].T)

# %% subspace overlap bar graph
    
# Or1 = np.array([Overlap[0,1,0],Overlap[0,2,0],)



# %% draw trajectories


from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Line3D
from matplotlib import cm
    
def draw_traj(traj,f,v,trmax,sc,g):
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
        if tr ==4:
            lc = Line3DCollection(segments, color = "red", linestyle = 'dotted')
        if tr ==3:
            lc = Line3DCollection(segments, color = "red", linestyle = 'dotted',alpha = 0)
        # elif tr <4:
        #     lc = Line3DCollection(segments, color = colors[tr], linestyle = 'solid')

        lc.set_array(time)
        lc.set_linewidth(4)
        ax.add_collection3d(lc)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        
        
        for m in [0]:
            ax.scatter(x[m], y[m], z[m], marker='o', color = "yellow")
        if tr == 1:
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

def draw_traj2(traj,v,trmax,sc,g):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    # styles = ['solid', 'dotted', 'solid','dotted']
    # cmap_names = ['autumn','autumn','winter','winter']
    styles = ['dotted','solid']
    cmap_names = ['autumn','winter','winter']
    for tr in np.arange(trmax):
        for ind in [0,1]:
            x = traj[ind][tr][:,0]
            y = traj[ind][tr][:,1]
            z = traj[ind][tr][:,2]
            
            x = ndimage.gaussian_filter(x,1)
            y = ndimage.gaussian_filter(y,1)
            z = ndimage.gaussian_filter(z,1)            
                
            time = np.arange(len(x))
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        
            colors = cm.copper(np.linspace(0,1,trmax))
            
            # norm = BoundaryNorm([0,19,29,49,89,109],cmap.N)
            # lc = Line3DCollection(segments, cmap=cmap_names[tr], norm=norm,linestyle = styles[tr])
            lc = Line3DCollection(segments, color = colors[tr],alpha = 0.5,linestyle = styles[ind])#linestyle = styles[tr])
            # lc = Line3D(x,y,z, markevery = [0,19,29,49,89,109], color = colors[tr])#linestyle = styles[tr])
            if tr ==4:
                lc = Line3DCollection(segments, color = "red", linestyle = styles[ind])
            if tr ==3:
                lc = Line3DCollection(segments, color = "red", linestyle = 'dotted',alpha = 0)
            # elif tr <4:
            #     lc = Line3DCollection(segments, color = colors[tr], linestyle = 'solid')
    
            lc.set_array(time)
            lc.set_linewidth(2)
            ax.add_collection3d(lc)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
            
            for m in [0]:
                ax.scatter(x[m], y[m], z[m], marker='o', color = "yellow")
            if tr == 0:
                ax.auto_scale_xyz(x,y,z)
                
    for tr in [trmax, trmax+1]:
        for ind in [0,1]:
            x = traj[ind][tr][:,0]
            y = traj[ind][tr][:,1]
            z = traj[ind][tr][:,2]
            
            x = ndimage.gaussian_filter(x,1)
            y = ndimage.gaussian_filter(y,1)
            z = ndimage.gaussian_filter(z,1)            
                
            time = np.arange(len(x))
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        
            colors = cm.copper(np.linspace(0,1,2))
            
            # norm = BoundaryNorm([0,19,29,49,89,109],cmap.N)
            # lc = Line3DCollection(segments, cmap=cmap_names[tr], norm=norm,linestyle = styles[tr])
            lc = Line3DCollection(segments, color = colors[tr-trmax],alpha = 1,linestyle = styles[ind])#linestyle = styles[tr])
            # lc = Line3D(x,y,z, markevery = [0,19,29,49,89,109], color = colors[tr])#linestyle = styles[tr])
            if tr ==4:
                lc = Line3DCollection(segments, color = "red", linestyle = styles[ind])
            if tr ==3:
                lc = Line3DCollection(segments, color = "red", linestyle = 'dotted',alpha = 0)
            # elif tr <4:
            #     lc = Line3DCollection(segments, color = colors[tr], linestyle = 'solid')
    
            lc.set_array(time)
            lc.set_linewidth(4)
            ax.add_collection3d(lc)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
            
            for m in [0]:
                ax.scatter(x[m], y[m], z[m], marker='o', color = "yellow")
            if tr == g:
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
            plt.gcf().set_size_inches(10, 10)
            # plt.savefig('charts/fig_size.png', )
            plt.savefig('Images/img' + str(n).zfill(3) + '.png',
                        dpi=200)



# %% trajectories


traj = {};

# D 0 is no Go, 1 is Go, 2 is all 

sm = 5
t1 = 0
t2 = 160
g = 8
traj[0] = {}
traj[1] = {}
R = {}

trmax = 10

# for tr in np.arange(trmax):
#     # R[tr]= np.concatenate((ndimage.gaussian_filter(D[0,tr][d_list2,t1:t2].T,[sm,0]),ndimage.gaussian_filter(D[1,tr][d_list2,t1:t2].T,[sm,0])),1)
#     # R[tr] = R[tr]-np.mean(R[tr][0:30,:],0)
#     R[tr] =ndimage.gaussian_filter(D[1,tr][d_list2,:].T,[sm,0])
#     traj[0][tr] = np.dot(R[tr],pca[g].components_.T)
    
for tr in np.arange(trmax):
    # R[tr]= np.concatenate((ndimage.gaussian_filter(D[0,tr][d_list2,t1:t2].T,[sm,0]),ndimage.gaussian_filter(D[1,tr][d_list2,t1:t2].T,[sm,0])),1)
    # R[tr] = R[tr]-np.mean(R[tr][0:30,:],0)
    for ind in [0,1]:
        n = np.sum(d_list2)
        # W = pca[g].components_[]
        traj[ind][tr] = np.dot(ndimage.gaussian_filter(D[ind,tr][d_list2,:].T,[sm,0]),pca[g].components_.T)
    
# for tr in np.arange(trmax):
#     # R[tr]= np.concatenate((ndimage.gaussian_filter(D[0,tr][d_list2,t1:t2].T,[sm,0]),ndimage.gaussian_filter(D[1,tr][d_list2,t1:t2].T,[sm,0])),1)
#     R[tr] = (ndimage.gaussian_filter(D_r[tr],[0,sm]))
#     # R[tr] = R[tr]-np.mean(R[tr][:,10:30])

#     # R[tr] = R[tr]-np.mean(R[tr][0:30,:],0)
#     traj[0][tr] = np.dot(R[tr].T,raster_model.Usv)
    

9


draw_traj2(traj,0,trmax-2,0,g)

# %% calculate euclidean distance between trajectories
trmax = 7
ED = np.zeros((trmax,trmax))
ind =1
for tr1 in np.arange(trmax):
    for tr2 in np.arange(trmax):
        ED[tr1,tr2] = np.linalg.norm(traj[ind][tr1]-traj[ind][tr2])
        
fig, axes = plt.subplots(1,1,figsize = (10,10))

# ED = ED/np.max(ED)
imshowobj = axes.imshow(ED,cmap = "hot")
# imshowobj.set_clim(0.3, 0.9) #correct
plt.colorbar(imshowobj) #adjusts scale to value range, looks OK     
# %% # %%

plt.close()
# draw_traj(traj,3,1,8,0)
import IPython.display as IPdisplay
import glob
from PIL import Image as PIL_Image
import imageio

images = [PIL_Image.open(image) for image in glob.glob('images/*.png')]
file_path_name = 'images/GLM_kernel/PIC_trajectory.gif'
imageio.mimsave(file_path_name, images)

# %%



from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Line3D
from matplotlib import cm
    
def draw_traj3(traj,v,sc,g):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    styles = ['solid', 'dotted', 'solid','dotted','solid','dotted']
    # cmap_names = ['autumn','autumn','winter','winter']
    # styles = ['solid','solid','dotted']
    cmap_names = ['autumn','winter','winter']
    for ind in [0,1]:
        x = traj[ind][g][:,0]
        y = traj[ind][g][:,1]
        z = traj[ind][g][:,2]
        
        x = ndimage.gaussian_filter(x,1)
        y = ndimage.gaussian_filter(y,1)
        z = ndimage.gaussian_filter(z,1)            
            
        time = np.arange(len(x))
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
        
    
        
        # norm = plt.Normalize(time.min(), time.max())
        # cmap=plt.get_cmap(cmap_names[tr])
        # colors=[cmap(float(ii)/(n-1)) for ii in range(np.size(segments,0))]
        # colors = cm.copper(np.linspace(0,1,trmax))
        
        # norm = BoundaryNorm([0,19,29,49,89,109],cmap.N)
        # lc = Line3DCollection(segments, cmap=cmap_names[tr], norm=norm,linestyle = styles[tr])
        if ind == 0:
            lc = Line3DCollection(segments, color = 'red',linestyle = styles[tr])
        # lc = Line3D(x,y,z, markevery = [0,19,29,49,89,109], color = colors[tr])#linestyle = styles[tr])
        if ind  ==1:
            lc = Line3DCollection(segments, color = "blue", linestyle = styles[tr])
        # elif tr <4:
        #     lc = Line3DCollection(segments, color = colors[tr], linestyle = 'solid')

        lc.set_array(time)
        lc.set_linewidth(4)
        ax.add_collection3d(lc)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        
        
        for m in [0]:
            ax.scatter(x[m], y[m], z[m], marker='o', color = "yellow")
        if ind == 1:
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

# %%

test = {}
traj = {}
traj[0] = {}
traj[1] = {}
for g in np.arange(8):
    test[g] = pca[g].fit_transform(ndimage.gaussian_filter(R[g],[1,0]))    
    traj[1][g] = test[g][0:160,:] - np.mean(test[g][10:30,:],0)
    traj[0][g] = test[g][160:,:]  - np.mean(test[g][170:200,:],0)

g = 0
draw_traj3(traj,0,0,g)


# %% 





pca = {}
max_k = 20;
# d_list = np.logical_and(good_list > 179, good_list < 600)
d_list = good_list > 179

d_list3 = good_list <= 179


d_list2 = d_list3
# d_list2 = good_list>-1

sm = 4
R = {}

R[0] = ndimage.gaussian_filter(D[0,8][d_list2,t1:t2].T,[sm,0])
R[1] = ndimage.gaussian_filter(D[1,8][d_list2,t1:t2].T,[sm,0])
R[4] = ndimage.gaussian_filter(D[0,9][d_list2,t1:t2].T,[sm,0])
R[5] = ndimage.gaussian_filter(D[1,9][d_list2,t1:t2].T,[sm,0])
R[2] = ndimage.gaussian_filter(D[0,4][d_list2,t1:t2].T,[sm,0])
R[3] = ndimage.gaussian_filter(D[1,4][d_list2,t1:t2].T,[sm,0])

# fig, axs = plt.subplots(1,6,figsize = (20,10))

pca = PCA(n_components=max_k)
RT = ndimage.gaussian_filter(D[2,8][d_list2,:].T,[sm,0])
    # R = ndimage.gaussian_filter(D_r[g][:,:].T,[sm,0])
    # R = ndimage.gaussian_filter(zscore(D[1,g][d_list2,:], axis = 1),[sm,0])
test = pca.fit_transform(ndimage.gaussian_filter(RT,[1,0]))        
test = test.T
# for t in range(0,5):
#     axs[t].plot(test[t,:], linewidth = 4)
#     axs[5].plot(np.cumsum(pca.explained_variance_ratio_), linewidth = 4)
    
    
# fig,axs = plt.subplots(1,1, figsize = (10,10))
# axs.plot(np.mean(R[0],1))


trmax = 6
for ind in np.arange(trmax):
    R[ind] = R[ind]- np.mean(R[ind][:30,:],0)
    traj[ind] = np.dot(R[ind],pca.components_.T)

# %% trajectory
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
styles = ['solid', 'dotted', 'solid','dotted','solid','dotted']
# cmap_names = ['autumn','autumn','winter','winter']
# styles = ['solid','solid','dotted']
cmap_names = ['autumn','winter','winter']

for tr in np.arange(trmax):
    x = traj[tr][:,0]
    y = traj[tr][:,1]
    z = traj[tr][:,2]
    
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
    lc = Line3DCollection(segments, color = colors[tr],linestyle = styles[tr])
    # lc = Line3D(x,y,z, markevery = [0,19,29,49,89,109], color = colors[tr])#linestyle = styles[tr])
    if tr ==2 or tr == 3:
        lc = Line3DCollection(segments, color = "red", linestyle = styles[tr])
    # elif tr <4:
    #     lc = Line3DCollection(segments, color = colors[tr], linestyle = 'solid')

    lc.set_array(time)
    lc.set_linewidth(4)
    ax.add_collection3d(lc)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    for m in [0]:
        ax.scatter(x[m], y[m], z[m], marker='o', color = "black")
    if tr == 1:
        ax.auto_scale_xyz(x,y,z)




# %% 

def visualize_components(component1, component2, labels, show=True):
  """
  Plots a 2D representation of the data for visualization with categories
  labelled as different colors.

  Args:
    component1 (numpy array of floats) : Vector of component 1 scores
    component2 (numpy array of floats) : Vector of component 2 scores
    labels (numpy array of floats)     : Vector corresponding to categories of
                                         samples

  Returns:
    Nothing.

  """

  plt.figure()
  
  plt.scatter(x=component1, y=component2, c=labels[0,:], alpha = (labels[1,:]+1)/2 , cmap='autumn')
  plt.xlabel('Component 1')
  plt.ylabel('Component 2')
  plt.colorbar(ticks=range(8))
  plt.clim(-0.5, 7.5)
  if show:
    plt.show()
# %% T-sne clustering
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
# D = D_cv[cv]
# Get images
# mnist = fetch_openml(name='mnist_784', as_frame=False, parser='auto')
# X_all = mnist.data
# labels_all = np.array([int(k) for k in mnist.target])


d_list = (good_list > 195)
d_list3 = good_list <= 195

d_list2 = d_list3
sm = 5
# R =  np.empty( shape=(160,0) )
R =  np.empty( shape=(0,np.sum(d_list2)) )
label = np.empty( shape=(2,0) )
tr_max = 8
for tr in np.arange(tr_max):
    if tr != 3:
        for ind in [0,1]:
            R = np.concatenate((R,ndimage.gaussian_filter(D[ind,tr][d_list2,:].T,[sm,0])),0)
            lbl = np.concatenate((np.ones((1,np.sum(d_list2)))*tr,np.ones((1,np.sum(d_list2)))*ind),0)
            label = np.concatenate((label,lbl),1)


# fig, axs = plt.subplots(1,6,figsize = (20,10))

# RT = np.dot(R,raster_model.Usv)

pca = PCA(n_components=64)
test = pca.fit_transform(R)        
# test = test.T
# # for t in range(0,5):
# #     axs[t].plot(test[t,:], linewidth = 4)
# #     axs[5].plot(np.cumsum(pca.explained_variance_ratio_), linewidth = 4)

# tsne_model = TSNE(n_components=30, perplexity=50, random_state=2020)
# embed = tsne_model.fit_transform(R.T)

# %%


from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Line3D
from matplotlib import cm


def draw_traj4(traj,v,trmax,sc,g):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    # styles = ['solid', 'dotted', 'solid','dotted']
    # cmap_names = ['autumn','autumn','winter','winter']
    styles = ['solid','solid']
    cmap_names = ['autumn','winter','winter']
    for tr in np.arange(trmax):
        for ind in [0,1]:
            x = traj[ind][tr][:,0]
            y = traj[ind][tr][:,1]
            z = traj[ind][tr][:,2]
            
            x = ndimage.gaussian_filter(x,1)
            y = ndimage.gaussian_filter(y,1)
            z = ndimage.gaussian_filter(z,1)            
                
            time = np.arange(len(x))
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
            if ind == 0:
                colors = cm.autumn(np.linspace(0,1,trmax))
            elif ind == 1:
                colors = cm.winter(np.linspace(0,1,trmax))
            
            # norm = BoundaryNorm([0,19,29,49,89,109],cmap.N)
            # lc = Line3DCollection(segments, cmap=cmap_names[tr], norm=norm,linestyle = styles[tr])
            lc = Line3DCollection(segments, color = colors[tr],alpha = 0.5,linestyle = styles[ind])#linestyle = styles[tr])
            # lc = Line3D(x,y,z, markevery = [0,19,29,49,89,109], color = colors[tr])#linestyle = styles[tr])
            if tr ==3:
                lc = Line3DCollection(segments, color = "purple", linestyle = styles[ind])
            # if tr ==3:
            #     lc = Line3DCollection(segments, color = "red", linestyle = 'dotted',alpha = 0)
            # elif tr <4:
            #     lc = Line3DCollection(segments, color = colors[tr], linestyle = 'solid')
    
            lc.set_array(time)
            lc.set_linewidth(2)
            ax.add_collection3d(lc)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
            
            for m in [0]:
                ax.scatter(x[m], y[m], z[m], marker='o', color = "black")
            # if tr == 0:
            #     ax.auto_scale_xyz(x,y,z)
    for tr in [trmax, trmax+1]:
        for ind in [0,1]:
            x = traj[ind][tr][:,0]
            y = traj[ind][tr][:,1]
            z = traj[ind][tr][:,2]
            
            x = ndimage.gaussian_filter(x,1)
            y = ndimage.gaussian_filter(y,1)
            z = ndimage.gaussian_filter(z,1)            
                
            time = np.arange(len(x))
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        
            colors = cm.copper(np.linspace(0,1,2))
            if ind == 0:
                colors = cm.autumn(np.linspace(0,1,2))
            elif ind == 1:
                colors = cm.winter(np.linspace(0,1,2))
            # norm = BoundaryNorm([0,19,29,49,89,109],cmap.N)
            # lc = Line3DCollection(segments, cmap=cmap_names[tr], norm=norm,linestyle = styles[tr])
            lc = Line3DCollection(segments, color = colors[tr-trmax],alpha = 1,linestyle = styles[ind])#linestyle = styles[tr])
    
            lc.set_array(time)
            lc.set_linewidth(4)
            ax.add_collection3d(lc)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
            
            # for m in [0]:
            #     ax.scatter(x[m], y[m], z[m], marker='o', color = "yellow")
                
            # for m in [158]:
            #     if ind == 0:
            #         ax.scatter(x[m], y[m], z[m], marker='o', color = "red")
            #     elif ind == 1:
            #         ax.scatter(x[m], y[m], z[m], marker='o', color = "blue")
                    
            # if tr == g and ind == 1:
            ax.auto_scale_xyz(x,y,z)
            # ax.set_xlim([-2,5])
            # ax.set_ylim([-3,3])
            # ax.set_zlim([-2,1])
            ax.set_xlim([-2,6])
            ax.set_ylim([-2,2])
            ax.set_zlim([-3,2])
                
                

            
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
            plt.gcf().set_size_inches(10, 10)
            # plt.savefig('charts/fig_size.png', )
            plt.savefig('Images/img' + str(n).zfill(3) + '.png',
                        dpi=200)


# %% 
traj = {}
traj[0] = {}
traj[1] = {}
R = {}

trmax = 7

# for tr in np.arange(trmax):
#     # R[tr]= np.concatenate((ndimage.gaussian_filter(D[0,tr][d_list2,t1:t2].T,[sm,0]),ndimage.gaussian_filter(D[1,tr][d_list2,t1:t2].T,[sm,0])),1)
#     # R[tr] = R[tr]-np.mean(R[tr][0:30,:],0)
#     R[tr] =ndimage.gaussian_filter(D[1,tr][d_list2,:].T,[sm,0])
#     traj[0][tr] = np.dot(R[tr],pca[g].components_.T)
m = 0 
sm = 5 
for tr in np.arange(trmax):
    # R[tr]= np.concatenate((ndimage.gaussian_filter(D[0,tr][d_list2,t1:t2].T,[sm,0]),ndimage.gaussian_filter(D[1,tr][d_list2,t1:t2].T,[sm,0])),1)
    # R[tr] = R[tr]-np.mean(R[tr][0:30,:],0)zz
    for ind in [0,1]:
        traj[ind][tr] = ndimage.gaussian_filter(test[m*160:(m+1)*160,:],[sm,0])
        traj[ind][tr] = traj[ind][tr]- np.mean(traj[ind][tr][10:30,:],0)
        m += 1

for ind in [0,1]:
    traj[ind][trmax] = (traj[ind][0] + traj[ind][1] + traj[ind][2])/3
    traj[ind][trmax+1] = (traj[ind][4] + traj[ind][5] + traj[ind][6])/3



# trmax = 1
draw_traj4(traj,0,trmax,0,trmax)


# %%

EDtot = {}

EDmax = 0
for ind in [0,1]:
    for tp in np.arange(4):
        if EDmax < np.max(EDtot[ind,tp]):
            EDmax = np.max(EDtot[ind,tp])
            
            
            
# np.max(EDtot)
# %% calculate euclidean distance between trajectories
trmax = 7
ind =1
ED = np.zeros((trmax,trmax))

tp = 3
EDtot[ind,tp] = np.zeros((trmax,trmax))

t1 = 110
t2 = 140
for tr1 in np.arange(trmax):
    for tr2 in np.arange(trmax):
        ED[tr1,tr2] = np.linalg.norm(traj[ind][tr1][t1:t2,:5]-traj[ind][tr2][t1:t2,:5])
        
EDtot[ind,tp]  = ED

fig, axes = plt.subplots(1,1,figsize = (10,10))

# ED = ED/np.max(ED)
imshowobj = axes.imshow(ED,cmap = "hot")
# imshowobj = axes.imshow(Overlap[:,:,0],cmap = "hot_r")
# imshowobj.set_clim(5, 30) #correct
plt.colorbar(imshowobj) #adjusts scale to value range, looks OK



# %% trajectory distance, quantified

fig, axes = plt.subplots(1,1,figsize = (10,10))
# 
ind = 0
tp = 3
ED = EDtot[ind,tp]/EDmax


# ED = ED/np.max(ED)
imshowobj = axes.imshow(ED,cmap = "hot")
imshowobj.set_clim(0, 0.75) #correct
fig, axes = plt.subplots(1,1,figsize = (10,10))

TD = {}
TD[0] = [ED[0,1],ED[0,2], ED[1,2],ED[4,5],ED[4,6], ED[5,6]]
TD[1] = [ED[0,4],ED[0,5], ED[0,6],ED[1,4],ED[1,5], ED[1,6],ED[2,4],ED[2,5], ED[2,6]]
TD[2] = [ED[3,0],ED[3,1], ED[3,2]]
TD[3] = [ED[3,4],ED[3,5], ED[3,6]]

mean_TD = np.zeros((1,4))
std_TD = np.zeros((1,4))
for t in np.arange(4):
    mean_TD[0,t] = np.mean(TD[t])
    std_TD[0,t] = np.std(TD[t]) #/np.sqrt(np.size(TD[t],0))

axes.bar(np.arange(4),mean_TD[0,:])
axes.errorbar(np.arange(4),mean_TD[0,:],std_TD[0,:], ecolor = 'black', elinewidth = 2, barsabove = True,linewidth = 0)
axes.set_ylim([0.05,1])


# %% calculate trajectory length



# %% for each segment, calculate stuff


trmax = 7
ED = np.zeros((trmax,trmax,160))
ind =0
for tr1 in np.arange(trmax):
    for tr2 in np.arange(trmax):
        # ED[tr1,tr2] = np.linalg.norm(traj[ind][tr1][:160,:10]-traj[ind][tr2][:160,:10])
        ED[tr1,tr2,:] = np.linalg.norm(traj[ind][tr1][:,:3]-traj[ind][tr2][:,:3],axis = 1)
        # ED[tr1,tr2] = np.linalg.norm(traj[1][tr1][60:90,:3]-traj[1][tr1][0:30,:3])
        
# fig, axes = plt.subplots(1,1,figsize = (10,10))

# # ED = ED/np.max(ED)
# imshowobj = axes.imshow(ED,cmap = "hot")

traj_sd = np.std(ED,axis = (0,1))

fig, axes = plt.subplots(1,1,figsize = (10,10))
plt.plot(ndimage.gaussian_filter(traj_sd, 2))
# %% re calculate subspace

n_cv = 20   
trmax = 8


Overlap = np.zeros((trmax-1,trmax-1)); # PPC_IC


k1 = 5
k2 = 19


W = {};
R = {};
fig, axes = plt.subplots(1,1,figsize = (10,10))

for g1 in [0,1,2,4,5,6,7]: #np.arange(trmax):
    R[g1] = np.concatenate((ndimage.gaussian_filter(D[1,g1][d_list2,:].T,[sm,0]),ndimage.gaussian_filter(D[0,g1][d_list2,:].T,[sm,0])),0)
    W[g1] = np.dot(R[g1].T,np.concatenate((traj[0][g1],traj[1][g1])))
    W[g1] = W[g1].T

for g1 in [0,1,2,4,5,6,7]: #np.arange(trmax):
    # R[g1] = np.concatenate((ndimage.gaussian_filter(D[1,g1][d_list2,:].T,[sm,0]),ndimage.gaussian_filter(D[0,g1][d_list2,:].T,[sm,0])),0)
    # W[g1] = np.dot(R[g1].T,np.concatenate((traj[0][g1],traj[1][g1])))
    # W[g1] = W[g1].T
    for g2 in [0,1,2,4,5,6,7]: # np.arange(trmax):
       S_value = np.zeros((1,k1))
       
       
       for d in np.arange(0,k1):
           S_value[0,d] = np.abs(np.dot(W[g1][d,:], W[g2][d,:].T))
           S_value[0,d] = S_value[0,d]/(np.linalg.norm(W[g1][d,:])*np.linalg.norm(W[g2][d,:]))
       if g1 < 4:
           if g2 < 4:
               Overlap[g1,g2] = np.max(S_value)
           elif g2 >= 4:
               Overlap[g1,g2-1] = np.max(S_value)
       elif g1 >= 4:
           if g2 < 4:
               Overlap[g1-1,g2] = np.max(S_value)
           elif g2 >= 4:
               Overlap[g1-1,g2-1] = np.max(S_value)
           
        
imshowobj = axes.imshow(Overlap[:,:],cmap = "hot_r")
imshowobj.set_clim(0.1, 0.9) #correct
plt.colorbar(imshowobj) #adjusts scale to value range, looks OK

# %% rastermapv2
# spks is neurons by time
# spks = np.load("spks.npy").astype("float32")
# spks = zscore(spks, axis=1)

# spks = zscore(D_r[0],axis = 1)
spks = R.T
# fit rastermap
# note that D_r is already normalized
model = Rastermap(n_PCs=64,
                  locality=0.75,
                  time_lag_window=5,
                  n_clusters = 50,
                  grid_upsample=5, keep_norm_X = False).fit(spks)
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
X_embedding = zscore(utils.bin1d(spks[isort], bin_size=25, axis=0), axis=1)


fig, ax = plt.subplots(figsize = (50,10))
D_hat = np.dot(model.Usv,model.Vsv.T)
ax.imshow(zscore(D_hat[isort, :],axis = 1), cmap="gray_r", vmin=0, vmax=1.2, aspect="auto")
# ax.imshow(D_hat)
# 

# np.size(D_hat,0)

tsne_model = TSNE(n_components=2, perplexity=100, random_state=100)
embed = tsne_model.fit_transform(D_hat)
# %%
# Visualize the data
with plt.xkcd():
  visualize_components(embed[:, 0], embed[:, 1], label)





# d_list2 = d_list3
# sm = 4

# g = trmax+1
# tsne = {}
# D_embedded = {}
# tsne[g] = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=100)
# R = ndimage.gaussian_filter(D[2,8][d_list2,:].T,[sm,0])

# D_embedded[g] =tsne[g].fit_transform(R.T)

# D_embedded[g].shape

# fig,ax = plt.subplots(1,1,figsize = (15,15))
# ax.scatter(D_embedded[g][:,0], D_embedded[g][:,1])





    
    
    