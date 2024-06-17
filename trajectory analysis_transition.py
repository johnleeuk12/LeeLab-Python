# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:42:52 2023

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
import seaborn as sns
from os.path import join as pjoin
from numba import jit, cuda


# from matplotlib.colors import BoundaryNorm

# from sklearn.metrics import get_scorer_names, d2_tweedie_score, make_scorer


# %% File name and directory

# change fname for filename
fname = 'CaData_all_all_session_v2_corrected.mat'

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

def import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind):    
    
    L_all = np.zeros((1,int(np.floor(D_ppc[n,3][0,max(D_ppc[n,2][:,0])]*1e3))+t_period+100))
    N_trial = np.size(D_ppc[n,2],0)
    
    # extracting licks, the same way
    for l in np.floor(D_ppc[n,1]*1e3):
        l = int(l) 
        if l < np.size(L_all,1):
            L_all[0,l-1] = 1 
    
    L = [] #np.zeros((N_trial,t_period+prestim))
    # for tr in range(N_trial):
    #     stim_onset = int(np.round(D_ppc[n,3][0,D_ppc[n,2][tr,0]]*1e3))
    #     lick_onset = int(np.round(D_ppc[n,3][0,D_ppc[n,2][tr,3]]*1e3))
    #     lick_onset = lick_onset-stim_onset
    #     L[tr,:] = L_all[0,stim_onset-prestim-1:stim_onset+t_period-1]
        
        # reformatting lick rates
    L2 = []
    # for w in range(int((t_period+prestim)/window)):
    #     l = np.sum(L[:,range(window*w,window*(w+1))],1)
    #     L2 = np.concatenate((L2,l)) 
            
    # L2 = np.reshape(L2,(int((t_period+prestim)/window),N_trial)).T


    X = D_ppc[n,2][:,2:6] # task variables
    Rt =  D_ppc[n,6] # reward time relative to stim onset, in seconds
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
    # stim_reward = 3500
    # total_period = t_period + stim_reward
    # Y = np.zeros((N_trial,int(((total_period))/window)))
    # # Y[0,:] = np.concatenate((Yraw[0,D_ppc[n,2][0,0]-1: D_ppc[n,2][0,0] + int(total_period/window)-1],
    # #                         Yraw[0,D_ppc[n,2][0,0]-1: D_ppc[n,2][0,0] + int(total_period/window)-1]))
    # Y[0,:] = Yraw[0,D_ppc[n,2][0,0]-1: D_ppc[n,2][0,0] + int(total_period/window)-1]
    # for tr in range(1,N_trial):
        
    #     Y[tr,:] = np.concatenate((Yraw[0,D_ppc[n,2][tr-1,0]-1 : D_ppc[n,2][tr-1,0] + int(t_period/window)-1], 
    #                                    Yraw[0,D_ppc[n,2][tr,0]-1 : D_ppc[n,2][tr,0] + int(stim_reward/window)-1]))
    #     # Y[tr,:] = np.concatenate((Yraw[0,D_ppc[n,2][tr-1,0]-1 : D_ppc[n,2][tr-1,0] + int(total_period/window)-1],
    #                               # Yraw[0,D_ppc[n,2][tr,0]-1 : D_ppc[n,2][tr,0] + int(total_period/window)-1]))

    
    
    # for t in np.arange(int(t_period/window)):
    #     Y[:,t] = Y[:,t]- np/median(Y[:,t])


                
    # select analysis and model parameters with c_ind
    
    if c_ind == 0:             
    # remove conditioning trials 
        Y = np.concatenate((Y[0:200,:],Y[D_ppc[n,4][0][0]:,:]),0)
        X = np.concatenate((X[0:200,:],X[D_ppc[n,4][0][0]:,:]),0)
        L2 = np.concatenate((L2[0:200,:],L2[D_ppc[n,4][0][0]:,:]),0)
    elif c_ind == 3:
    # only contain conditioning trials    
        # Y = Y[201:D_ppc[n,4][0][0]]
        # X = X[201:D_ppc[n,4][0][0]]
        # L2 = L2[201:D_ppc[n,4][0][0]]
        # Y = Y[201:250]
        # X = X[201:250]
        # L2 = L2[201:250]
        c1 = 200
        c2 = D_ppc[n,4][0][0] + 25
        Y = Y[c1:c2]
        X = X[c1:c2]
        L2 = L2[c1:c2]
    elif c_ind == 1:
        c1 = 0
        c2 = 200
        Y = Y[c1:c2]
        X = X[c1:c2]
        L2 = L2[c1:c2]
    elif c_ind == 2:
        Y = Y[D_ppc[n,4][0][0]:,:]
        X = X[D_ppc[n,4][0][0]:,:]
        # L2 = L2[D_ppc[n,4][0][0]:,:]
        

    
    # Add reward  history
    Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
    Xpre = Xpre[:,None]
    
    # 10/05 adding stim history, inverting sign for 10kHz = Go
    Xprestim = np.concatenate(([0],-1*X[0:-1,3]+1),0)
    Xprestim = Xprestim[:,None]
    # Add reward instead of action
    # currently it's Contingency, Stimulus, Reward, and Reward History
    # # X2 = np.column_stack([X[:,0],X[:,3],
    # #                       X[:,2]*X[:,1],Xpre]) 

    X2 = np.column_stack([Xprestim,-1*X[:,3]+1,
                          X[:,2]*X[:,1],Xpre]) 
    # XHit = X[:,1]*(-1*X[:,3]+1)
    # XFA = X[:,1]*X[:,3]
    # Xmiss = (-1*X[:,1]+1)*(-1*X[:,3]+1)
    # XCR = (-1*X[:,1]+1)*X[:,3]
    
    # X2 =   np.column_stack([XHit,Xmiss,XFA,XCR])
    # X2 = np.row_stack([[0,0,0,0],X2[:-1,:]])

    
    
    # testing Hit, error etc
    # Adding Lick(action as well)
    # X2 = np.column_stack([Xprestim,-1*X[:,3]+1,X[:,1],
    #                       X[:,2]*X[:,1],Xpre]) 


    
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



D_ppc = load_matfile_Ca()
good_list = find_good_data_Ca(t_period)


# %% get neural data

lenx = 160 # Length of data, 8000ms, with a 50 ms window.
# good_list = np.arange(336)
# good_list = good_listRU
D_all = np.zeros((len(good_list),lenx))
D = {}
tr = 0
D[0,tr] = np.zeros((len(good_list),lenx))
D[1,tr] = np.zeros((len(good_list),lenx))


c_ind = 3
Y = {}
Ygo = {}
m = 0
for n in good_list: 
    Ygo[m] = []
    n = int(n)
    X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
    D_all[m,:] = np.mean(Y,0) #/(np.max(np.mean(Y,0)) + 0.5) # Soft normalisation, alpha = 0.5
    D[0,tr][m,:] = np.mean(Y[X[:,1] == 0,:],0)-np.mean(Y[X[:,1] == 0,:30])#/(np.max(np.mean(Y,0)) + 0.5)
    D[1,tr][m,:] = np.mean(Y[X[:,1] == 1,:],0)-np.mean(Y[X[:,1] == 1,:30])#/(np.max(np.mean(Y,0)) + 0.5)
    m += 1



# %% Plot Go vs No-Go


fig, axes = plt.subplots(figsize =(10,10))

cmap = ["tab:blue","tab:red"]
xaxis = np.linspace(-2,6,lenx+1)
xaxis  = xaxis[1:]
for c in [0,1]:
    gD = ndimage.gaussian_filter(D[c,0][133:,:],[0,3])
    sD = np.std(gD,0)/np.sqrt(np.size(gD,0))
    axes.plot(xaxis,np.mean(gD,0),color = cmap[c],linewidth = 4)
    axes.fill_between(xaxis,np.mean(gD,0)+sD,np.mean(gD,0)-sD,color = cmap[c], alpha = 0.3)
    # axes.set_ylim([-0.10, 1.05])
    axes.set_ylim([-0.15, 0.40])










