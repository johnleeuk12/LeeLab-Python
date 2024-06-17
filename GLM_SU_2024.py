# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:28:11 2024

GLM analysis, without segmenting data by trial onset time. 


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
import seaborn as sns
from os.path import join as pjoin
from numba import jit, cuda


# %% File name and directory

# change fname for filename
# fname = 'CaData_all_all_session_v2_corrected.mat'
fname = 'SUData_IC.mat'

fdir = 'D:\Python\Data'




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

def sliding_median(arr, window):
    
    return np.median(np.lib.stride_tricks.sliding_window_view(arr, (window,)), axis=1)

# @jit(target_backend='cuda')                         
def import_data_w_spikes(D_ppc,n,window,c_ind):
    D_ppc = load_matfile()
    timepoint = int(max(D_ppc[n,3][:,0])*1e3)+t_period+100;
    
    S_all = np.zeros((1,timepoint)) # spike time in seconds
    L_all = np.zeros((1,timepoint))
    N_trial = np.size(D_ppc[n,1],0)
    
    # extracting spikes from data
    for sp in np.array(D_ppc[n,0][:,0]):
        if np.floor(sp*1e3) < np.size(S_all,1):
            S_all[0,int(np.floor(sp*1e3))-1] = 1  #spike time starts at 1 but indexing starts at 0
                
    # Y = S_all
    
    time_ind = np.arange(0,np.size(S_all,1),window)
    Y = np.zeros((1,len(time_ind)-1))   
    for t in np.arange(len(time_ind)-1):
        Y[0,t] = np.mean(S_all[0,int(time_ind[t]):int(time_ind[t+1])])*1e3

    # extracting spikes, end
    
    ### Extract Lick ### 
    L_all = np.zeros((1,len(time_ind)-1))
    L_all_onset = np.zeros((1,len(time_ind)-1))
    L_all_offset = np.zeros((1,len(time_ind)-1))
    # Rt = np.zeros((1,len(time_ind)-1))
    Ln = np.array(D_ppc[n,1][:,0])
    InterL = Ln[1:]- Ln[:-1]
    lick_onset= np.where(InterL>2)[0] # lick bout boundary =2
    lick_onset = lick_onset+1
    lick_offset = lick_onset-1
    
    for l in np.floor(D_ppc[n,1]*(1e3/window)): 
        L_all[0,int(l[0])-1] = 1 
            
    for l in np.floor(Ln[lick_onset]*(1e3/window)):
        L_all_onset[0,int(l)-1] = 1
    
    for l in np.floor(Ln[lick_offset]*(1e3/window)):
        L_all_offset[0,int(l)-1] = 1 
              
    
    ### Extract Lick End ###
    
    stim_onset = np.round(D_ppc[n,3][:,0]*1e3)
    stim_onset = stim_onset.astype(int)
    if len(stim_onset) > 460:
        stim_onset = np.concatenate((stim_onset[:200],stim_onset[260:460]))
    else:
        stim_onset = np.concatenate((stim_onset[:200],stim_onset[260:]))    
    # Rt = np.floor(D_ppc[n,6][:,0]*(1e3/window))
    # Rt =Rt.astype(int)
    
    # Lt = np.zeros((400,50))
    # for st in np.arange(400):
    #     onset =int(np.floor(stim_onset[st]/window))
    #     Lt[st,:] = L_all[0,onset:onset+50]
    

    # temporary Rt, reward window if hit, 0 else
    Rt = np.zeros_like(stim_onset)
    Rt[:] = stim_onset[:]+1600
    X = D_ppc[n,2] # task variables
    if np.size(X,0) >400:
        X = X[:400,:]
    
    
    Rt[X[:,2]==0] = 0
    # stim_onset = np.round(D_ppc[n,3][:,0]*1e3)
    # stim_onset = stim_onset.astype(int)
    # if len(stim_onset) > 460:
    #     stim_onset = np.concatenate((stim_onset[:200],stim_onset[260:460]))
    # else:
    #     stim_onset = np.concatenate((stim_onset[:200],stim_onset[260:])) 
    stim_onset = np.floor(stim_onset/window).astype(int)
    Rt = np.floor(Rt/window).astype(int)


    Xstim = X[:,0]

    XHit = X[:,2]
    XFA  = X[:,3]
    Xmiss = X[:,4]
    XCR = X[:,5]
    
    ### Create variables ###
    ED1 = int(500/window) # 500ms pre, 1second post lag
    ED2 = int(1000/window)
    stim_dur = int(500/window) # 500ms stim duration
    delay = int(1000/window) # 1 second delay
    r_dur = int(1000/window) # 2 second reward duration 
    ED3 = int(7000/window) # 4 seconds post reward lag
    ED_hist1 = int(3000/window) # 4 seconds pre-stim next trial
    # ED_hist2 = int(1500/window) # 1.5 seconds post-stim next trial
    # h_dur = 500/window
    
    # ED1 = 5 # 500ms pre, 1second post lag
    # ED2 = 10
    # stim_dur = 5 # 500ms stim duration
    # delay = 10 # 1 second delay
    # r_dur = 10 # 2 second reward duration 
    # ED3 = 20 # 4 seconds post reward lag
    # ED_hist1 = 50 # 4 seconds pre-stim next trial
    # ED_hist2 = 15 # 1.5 seconds post-stim next trial
    # h_dur = 5
    
    
    X3_Lick_onset = np.zeros((ED1+ED2+1,np.size(Y,1)))
    X3_Lick_offset = np.zeros_like(X3_Lick_onset)
    
    X3_Lick_onset[0,:] = L_all_onset
    X3_Lick_offset[0,:] = L_all_offset

    for lag in np.arange(ED1):
        X3_Lick_onset[lag+1,:-lag-1] = L_all_onset[0,lag+1:]
        X3_Lick_offset[lag+1,:-lag-1] = L_all_offset[0,lag+1:]
    
    for lag in np.arange(ED2):
        X3_Lick_onset[lag+ED1+1,lag+1:] = L_all_onset[0,:-lag-1]
        X3_Lick_offset[lag+ED1+1,lag+1:] = L_all_offset[0,:-lag-1]
        
      
    X3_go = np.zeros((ED2+1,np.size(Y,1)))
    X3_ng = np.zeros_like(X3_go)
    
    for st in stim_onset[(Xstim == 1)]:
        X3_go[0,st:st+stim_dur] = 1
    
    for st in stim_onset[(Xstim ==0)]:
        X3_ng[0,st:st+stim_dur] = 1
        
    for lag in np.arange(ED2):
        X3_go[lag+1,lag+1:] = X3_go[0,:-lag-1]
        X3_ng[lag+1,lag+1:] = X3_ng[0,:-lag-1]
    
    
    X3_Hit = np.zeros((ED3+1,np.size(Y,1)))
    X3_FA = np.zeros_like(X3_Hit)
    X3_Miss = np.zeros_like(X3_Hit)
    X3_CR = np.zeros_like(X3_Hit)
    for r in Rt[(XHit == 1)]:
        if r != 0:
            r = r-20
            X3_Hit[0,r:r+r_dur] = 1
    
    for r in Rt[(XFA == 1)]:
        if r != 0:
            r = r-20
            X3_FA[0,r:r+r_dur] = 1

    
    for st in stim_onset[(Xmiss ==1)]:        
        X3_Miss[0,st+delay+stim_dur : st+delay+stim_dur+r_dur] = 1

    for st in stim_onset[(XCR ==1)]:        
        X3_CR[0,st+delay+stim_dur : st+delay+stim_dur+r_dur] = 1 
        
        
    for lag in np.arange(ED3):
        X3_Hit[lag+1,lag+1:] = X3_Hit[0,:-lag-1]
        X3_FA[lag+1,lag+1:] = X3_FA[0,:-lag-1]
        X3_Miss[lag+1,lag+1:] = X3_Miss[0,:-lag-1]
        X3_CR[lag+1,lag+1:] = X3_CR[0,:-lag-1]

    # X3_Hit_hist = np.zeros((ED_hist1+ED_hist2+1,np.size(Y,1)))
    # X3_FA_hist = np.zeros((ED_hist1+ED_hist2+1,np.size(Y,1)))
    
    X3_Hit_hist = np.zeros((ED_hist1+1,np.size(Y,1)))
    X3_FA_hist = np.zeros((ED_hist1++1,np.size(Y,1)))
    # XHit_prev = np.concatenate(([False], XHit[0:-1]), axis = 0)
    # XFA_prev = np.concatenate(([False], XFA[0:-1]), axis = 0)
    
    
    # X3_Hit_hist[0,20:] = X3_Hit[0,:-20]
    # X3_FA_hist[0,20:] = X3_FA[0,:-20]
    
    # for lag in np.arange(ED_hist1):
    #     X3_Hit_hist[lag+1,lag+1:] = X3_Hit_hist[0,:-lag-1]
    #     X3_FA_hist[lag+1,lag+1:] = X3_FA_hist[0,:-lag-1]

    X3 = {}
    X3[0] = X3_Lick_onset
    X3[1] = X3_Lick_offset
    X3[2] = X3_go
    X3[3] = X3_ng
    X3[4] = X3_Hit
    X3[5] = X3_FA
    X3[6] = X3_Miss
    X3[7] = X3_CR
    X3[8] = X3_Hit_hist
    X3[9] = X3_FA_hist
    
    if c_ind == 1:
        c1 = stim_onset[1]-100
        c2 = stim_onset[151]
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[1:150]
        r_onset = r_onset[1:150]
        Xstim = Xstim[1:150]
    
    elif c_ind == 3:
        c1 = stim_onset[200]-100
        c2 = stim_onset[D_ppc[n,4][0][0]+26] 
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[200:D_ppc[n,4][0][0]+26]
        r_onset = r_onset[200:D_ppc[n,4][0][0]+26]
        Xstim = Xstim[200:D_ppc[n,4][0][0]+26]

    elif c_ind == 2:       
        c1 = stim_onset[D_ppc[n,4][0][0]+26]-100 
        c2 = stim_onset[399]
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[D_ppc[n,4][0][0]+26:398]
        r_onset = r_onset[D_ppc[n,4][0][0]+26:398]
        Xstim = Xstim[D_ppc[n,4][0][0]+26:398]

        
    
    Y = Y[:,c1:c2]
    L_all = L_all[:,c1:c2]
    L_all_onset = L_all_onset[:,c1:c2]
    L_all_offset = L_all_offset[:,c1:c2]
    
    Y0 = np.mean(Y[:,:stim_onset[1]-50])
    
    for ind in np.arange(len(X3)):
        X3[ind] = X3[ind][:,c1:c2]         



    return X3,Y, L_all,L_all_onset, L_all_offset, stim_onset2, r_onset, Xstim,Y0    


def import_data_w_Ca(D_ppc,n,window,c_ind):    
    # For each neuron, get Y, neural data and X task variables.  
    # Stim onset is defined by stim onset time
    # Reward is defined by first lick during reward presentation
    # Lick onset, offset are defined by lick times
    # Hit vs FA are defined by trial conditions
    
    

    N_trial = np.size(D_ppc[n,2],0)
    
    # extracting licks, the same way
    

    
    
    
    ### Extract Ca trace ###
    Yraw = {}
    Yraw = D_ppc[n,0]
    time_point = D_ppc[n,3]*1e3
    t = 0
    time_ind = []
    while t*window < np.max(time_point):
        time_ind = np.concatenate((time_ind,np.argwhere(time_point[0,:]>t*window)[0]))
        t += 1
    
    
    Y = np.zeros((1,len(time_ind)-1))   
    for t in np.arange(len(time_ind)-1):
        Y[0,t] = np.mean(Yraw[0,int(time_ind[t]):int(time_ind[t+1])])
        
    
    ### Extract Lick ### 
    L_all = np.zeros((1,len(time_ind)-1))
    L_all_onset = np.zeros((1,len(time_ind)-1))
    L_all_offset = np.zeros((1,len(time_ind)-1))
    # Rt = np.zeros((1,len(time_ind)-1))
    Ln = np.array(D_ppc[n,1])
    InterL = Ln[1:,:]- Ln[:-1,:]
    lick_onset= np.where(InterL[:,0]>2)[0] # lick bout boundary =2
    lick_onset = lick_onset+1
    lick_offset = lick_onset-1
    
    for l in np.floor(D_ppc[n,1]*(1e3/window)): 
        L_all[0,int(l[0])-1] = 1 
            
    for l in np.floor(Ln[lick_onset,0]*(1e3/window)):
        L_all_onset[0,int(l)-1] = 1
    
    for l in np.floor(Ln[lick_offset,0]*(1e3/window)):
        L_all_offset[0,int(l)-1] = 1 
            
    # for l in np.floor(D_ppc[n,6][:,0]*(1e3/window)):
    #     Rt[0,int(l)-1] = 1     
    
    ### Extract Lick End ###
    

    stim_onset =np.round(D_ppc[n,3][0,D_ppc[n,2][:,0]]*(1e3/window))
    stim_onset = stim_onset.astype(int)
    Rt = np.floor(D_ppc[n,6][:,0]*(1e3/window))
    Rt =Rt.astype(int)

    X = D_ppc[n,2][:,2:6] # task variables
    Xstim = X[:,0]
    XHit = (X[:,0]==1)*(X[:,1]==1)
    XFA  = (X[:,0]==0)*(X[:,1]==1)
    Xmiss = (X[:,0]==1)*(X[:,1]==0)
    XCR = (X[:,0]==0)*(X[:,1]==0)
    
    
    ### Create variables ###
    ED1 = 5 # 500ms pre, 1second post lag
    ED2 = 10
    stim_dur = 5 # 500ms stim duration
    delay = 10 # 1 second delay
    r_dur = 10 # 2 second reward duration 
    ED3 = 20 # 4 seconds post reward lag
    ED_hist1 = 50 # 4 seconds pre-stim next trial
    ED_hist2 = 15 # 1.5 seconds post-stim next trial
    h_dur = 5
    
    X3_Lick_onset = np.zeros((ED1+ED2+1,np.size(Y,1)))
    X3_Lick_offset = np.zeros_like(X3_Lick_onset)
    
    X3_Lick_onset[0,:] = L_all_onset
    X3_Lick_offset[0,:] = L_all_offset

    for lag in np.arange(ED1):
        X3_Lick_onset[lag+1,:-lag-1] = L_all_onset[0,lag+1:]
        X3_Lick_offset[lag+1,:-lag-1] = L_all_offset[0,lag+1:]
    
    for lag in np.arange(ED2):
        X3_Lick_onset[lag+ED1+1,lag+1:] = L_all_onset[0,:-lag-1]
        X3_Lick_offset[lag+ED1+1,lag+1:] = L_all_offset[0,:-lag-1]
        
      
    X3_go = np.zeros((ED2+1,np.size(Y,1)))
    X3_ng = np.zeros_like(X3_go)
    
    for st in stim_onset[(Xstim == 1)]:
        X3_go[0,st:st+stim_dur] = 1
    
    for st in stim_onset[(Xstim ==0)]:
        X3_ng[0,st:st+stim_dur] = 1
        
    for lag in np.arange(ED2):
        X3_go[lag+1,lag+1:] = X3_go[0,:-lag-1]
        X3_ng[lag+1,lag+1:] = X3_ng[0,:-lag-1]
    
    
    X3_Hit = np.zeros((ED3+1,np.size(Y,1)))
    X3_FA = np.zeros_like(X3_Hit)
    X3_Miss = np.zeros_like(X3_Hit)
    X3_CR = np.zeros_like(X3_Hit)
    for r in Rt[(XHit == 1)]:
        if r != 0:
            X3_Hit[0,r:r+r_dur] = 1
    
    for r in Rt[(XFA == 1)]:
        if r != 0:
            X3_FA[0,r:r+r_dur] = 1
            
    for st in stim_onset[(Xmiss ==1)]:        
        X3_Miss[0,st+delay+stim_dur : st+delay+stim_dur+r_dur] = 1

    for st in stim_onset[(XCR ==1)]:        
        X3_CR[0,st+delay+stim_dur : st+delay+stim_dur+r_dur] = 1 
        
        
    for lag in np.arange(ED3):
        X3_Hit[lag+1,lag+1:] = X3_Hit[0,:-lag-1]
        X3_FA[lag+1,lag+1:] = X3_FA[0,:-lag-1]
        X3_Miss[lag+1,lag+1:] = X3_Miss[0,:-lag-1]
        X3_CR[lag+1,lag+1:] = X3_CR[0,:-lag-1]

    # X3_Hit_hist = np.zeros((ED_hist1+ED_hist2+1,np.size(Y,1)))
    # X3_FA_hist = np.zeros((ED_hist1+ED_hist2+1,np.size(Y,1)))
    
    X3_Hit_hist = np.zeros((ED_hist1+1,np.size(Y,1)))
    X3_FA_hist = np.zeros((ED_hist1++1,np.size(Y,1)))
    # XHit_prev = np.concatenate(([False], XHit[0:-1]), axis = 0)
    # XFA_prev = np.concatenate(([False], XFA[0:-1]), axis = 0)
    
    
    X3_Hit_hist[0,20:] = X3_Hit[0,:-20]
    X3_FA_hist[0,20:] = X3_FA[0,:-20]
    
    for lag in np.arange(ED_hist1):
        X3_Hit_hist[lag+1,lag+1:] = X3_Hit_hist[0,:-lag-1]
        X3_FA_hist[lag+1,lag+1:] = X3_FA_hist[0,:-lag-1]
    # for st in stim_onset[(XHit_prev ==1)]:
    #     X3_Hit_hist[0,st:st+h_dur] = 1
    # for st in stim_onset[(XFA_prev ==1)]:
    #     X3_FA_hist[0,st:st+h_dur] = 1 
    
    
    # for lag in np.arange(ED_hist1):
    #     X3_Hit_hist[lag+1,:-lag-1] = X3_Hit_hist[0,lag+1:]
    #     X3_FA_hist[lag+1,:-lag-1] = X3_FA_hist[0,lag+1:]
    
    # for lag in np.arange(ED_hist2):
    #     X3_Hit_hist[lag+ED_hist1+1,lag+1:] = X3_Hit_hist[0,:-lag-1]
    #     X3_FA_hist[lag+ED_hist1+1,lag+1:] = X3_FA_hist[0,:-lag-1]
    
    # for r in Rt[()]
    # gather X variables
    
    
    X3 = {}
    X3[0] = X3_Lick_onset
    X3[1] = X3_Lick_offset
    X3[2] = X3_go
    X3[3] = X3_ng
    X3[4] = X3_Hit
    X3[5] = X3_FA
    X3[6] = X3_Miss
    X3[7] = X3_CR
    X3[8] = X3_Hit_hist
    X3[9] = X3_FA_hist
    
    if c_ind == 1:
        c1 = stim_onset[1]-100
        c2 = stim_onset[151]
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[1:150]
        r_onset = r_onset[1:150]
        Xstim = Xstim[1:150]
    
    elif c_ind == 3:
        c1 = stim_onset[200]-100
        c2 = stim_onset[D_ppc[n,4][0][0]+26] 
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[200:D_ppc[n,4][0][0]+26]
        r_onset = r_onset[200:D_ppc[n,4][0][0]+26]
        Xstim = Xstim[200:D_ppc[n,4][0][0]+26]

    elif c_ind == 2:       
        c1 = stim_onset[D_ppc[n,4][0][0]+26]-100 
        c2 = stim_onset[399]
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[D_ppc[n,4][0][0]+26:398]
        r_onset = r_onset[D_ppc[n,4][0][0]+26:398]
        Xstim = Xstim[D_ppc[n,4][0][0]+26:398]

        

    Y = Y[:,c1:c2]
    L_all = L_all[:,c1:c2]
    L_all_onset = L_all_onset[:,c1:c2]
    L_all_offset = L_all_offset[:,c1:c2]
    
    Y0 = np.mean(Y[:,:stim_onset[1]-50])
    
    for ind in np.arange(len(X3)):
        X3[ind] = X3[ind][:,c1:c2]         



    return X3,Y, L_all,L_all_onset, L_all_offset, stim_onset2, r_onset, Xstim,Y0

# %% glm_per_neuron function code
def glm_per_neuron(n,c_ind, fig_on):
    X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim, Y0 = import_data_w_spikes(D_ppc,n,window,c_ind)
    

        
    
    # Y2 = Y
    Y2 = ndimage.gaussian_filter(Y,5) #-Y0
    # X4 = np.ones((1,np.size(Y)))
    
    reg = ElasticNet(alpha = 4*1e-2, l1_ratio = 0.5, fit_intercept=True) #Using a linear regression model with Ridge regression regulator set with alpha = 1
    ss= ShuffleSplit(n_splits=k, test_size=0.30, random_state=0)
    
    ### initial run, compare each TV ###
    Nvar= len(X)
    compare_score = {}
    int_alpha = 10
    for a in np.arange(Nvar+1):
        
        # X4 = np.ones_like(Y)*int_alpha
        X4 = np.zeros_like(Y)

        if a < Nvar:
            X4 = np.concatenate((X4,X[a]),axis = 0)

        cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                    return_estimator = True, 
                                    scoring = 'r2') 
        compare_score[a] = cv_results['test_score']
    
    f = np.zeros((1,Nvar))
    p = np.zeros((1,Nvar))
    score_mean = np.zeros((1,Nvar))
    for it in np.arange(Nvar):
        f[0,it], p[0,it] = stats.ks_2samp(compare_score[it],compare_score[Nvar],alternative = 'less')
        score_mean[0,it] = np.mean(compare_score[it])

    max_it = np.argmax(score_mean)
    init_score = compare_score[max_it]
    init_compare_score = compare_score
    
    if p[0,max_it] > 0.05:
            max_it = []
    else:  
            # === stepwise forward regression ===
            step = 0
            while step < Nvar:
                max_ind = {}
                compare_score2 = {}
                f = np.zeros((1,Nvar))
                p = np.zeros((1,Nvar))
                score_mean = np.zeros((1,Nvar))
                for it in np.arange(Nvar):
                    m_ind = np.unique(np.append(max_it,it))
                    # X4 = np.ones_like(Y)*int_alpha
                    X4 = np.zeros_like(Y)
                    for a in m_ind:
                        X4 = np.concatenate((X4,X[a]),axis = 0)

                    
                    cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                                return_estimator = True, 
                                                scoring = 'r2') 
                    compare_score2[it] = cv_results['test_score']
    
                    f[0,it], p[0,it] = stats.ks_2samp(compare_score2[it],init_score,alternative = 'less')
                    score_mean[0,it] = np.mean(compare_score2[it])
                max_ind = np.argmax(score_mean)
                if p[0,max_ind] > 0.05 or p[0,max_ind] == 0:
                    step = Nvar
                else:
                    max_it = np.unique(np.append(max_it,max_ind))
                    init_score = compare_score2[max_ind]
                    step += 1
                    
            # === forward regression end ===
            
            # === running regression with max_it ===
            
            # X4 = np.ones_like(Y)*int_alpha
            X4 = np.zeros_like(Y)
            if np.size(max_it) == 1:
                X4 = np.concatenate((X4,X[max_it]),axis = 0)
            else:
                for a in max_it:
                    X4 = np.concatenate((X4,X[a]),axis = 0)
            
            cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                        return_estimator = True, 
                                        scoring = 'r2') 
            score3 = cv_results['test_score']
            
            theta = [] 
            inter = []
            yhat = []
            for model in cv_results['estimator']:
                theta = np.concatenate([theta,model.coef_]) 
                # inter = np.concatenate([inter, model.intercept_])
                yhat =np.concatenate([yhat, model.predict(X4.T)])
                
            theta = np.reshape(theta,(k,-1)).T
            yhat = np.reshape(yhat,(k,-1)).T
            yhat = yhat + Y0
    
    TT = {}
    lg = 1
    
    if np.size(max_it) ==1:
        a = np.empty( shape=(0, 0) )
        max_it = np.append(a, [int(max_it)]).astype(int)
    try:
        for t in max_it:
            TT[t] = X[t].T@theta[lg:lg+np.size(X[t],0),:]  
            lg = lg+np.size(X[t],0)
    except: 
        TT[max_it] = X[max_it].T@theta[lg:lg+np.size(X[max_it],0),:]  
    
    if 4 in max_it:
        if 8 in max_it:
            TT[4] = TT[4] + TT[8]
    elif 8 in max_it:
        TT[4] = TT[8]
        max_it = np.append(max_it, [4])
    
        
    if 5 in max_it:
        if 9 in max_it:
            TT[5] = TT[5] + TT[9]
    elif 9 in max_it:
        TT[5] = TT[9]
        max_it = np.append(max_it, [5])
        
    
    # === figure === 
    if fig_on ==1:
        prestim = int(2000/window)
        t_period = int(6000/window)
        
        y = np.zeros((t_period+prestim,np.size(stim_onset)))
        yh = np.zeros((t_period+prestim,np.size(stim_onset)))
        l = np.zeros((t_period+prestim,np.size(stim_onset))) 
        weight = {}
        for a in np.arange(Nvar):
           weight[a] = np.zeros((t_period+prestim,np.size(stim_onset))) 
        
        yhat_mean = np.mean(yhat,1).T - Y0    
        for st in np.arange(np.size(stim_onset)):
            y[:,st] = Y2[0,stim_onset[st]-prestim: stim_onset[st]+t_period]
            yh[:,st] = yhat_mean[stim_onset[st]-prestim: stim_onset[st]+t_period]
            l[:,st] = Lm[0,stim_onset[st]-prestim: stim_onset[st]+t_period]
            # if np.size(max_it)>1:
            for t in max_it:
                weight[t][:,st] = np.mean(TT[t][stim_onset[st]-prestim: stim_onset[st]+t_period,:],1)
            # else:
            #     weight[max_it][:,st] = np.mean(TT[max_it][stim_onset[st]-prestim: stim_onset[st]+t_period,:],1)
            
    
        
        xaxis = np.arange(t_period+prestim)- prestim
        xaxis = xaxis*1e-1
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))        
        cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
        clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR','Hit_pre','FA_pre']
        lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted']
        
        ### plot y and y hat
        stim_ind1 = (Xstim ==1)
        stim_ind2 = (Xstim ==0)
    
        y1 = ndimage.gaussian_filter(np.mean(y[:,stim_ind1],1),0)
        y2 = ndimage.gaussian_filter(np.mean(y[:,stim_ind2],1),0)
        s1 = np.std(y[:,stim_ind1],1)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(y[:,stim_ind2],1)/np.sqrt(np.sum(stim_ind2))
        
        ax1.plot(xaxis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y1-s1, y1+s1, color = "blue",alpha = 0.5)
        ax1.plot(xaxis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y2-s2, y2+s2, color = "red",alpha = 0.5)
        
        y1 = ndimage.gaussian_filter(np.mean(yh[:,stim_ind1],1),0)
        y2 = ndimage.gaussian_filter(np.mean(yh[:,stim_ind2],1),0)
        s1 = np.std(yh[:,stim_ind1],1)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(yh[:,stim_ind2],1)/np.sqrt(np.sum(stim_ind2))
        
        ax1.plot(xaxis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y1-s1, y1+s1, color = "gray",alpha = 0.5)
        ax1.plot(xaxis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y2-s2, y2+s2, color = "gray",alpha = 0.5)
        
        
        
        ### plot model weights
        for a in np.arange(Nvar):
            y1 = ndimage.gaussian_filter(np.mean(weight[a],1),0)
            s1 = np.std(weight[a],1)/np.sqrt(np.size(weight[a],1))
            
            
            ax2.plot(xaxis,ndimage.gaussian_filter(y1,1),linewidth = 2.0,
                     color = cmap[a], label = clabels[a], linestyle = lstyles[a])
            ax2.fill_between(xaxis,(ndimage.gaussian_filter(y1,1) - s1),
                            (ndimage.gaussian_filter(y1,1)+ s1), color=cmap[a], alpha = 0.2)
        
        ### plot lick rate ###
        
        y1 = ndimage.gaussian_filter(np.mean(l[:,stim_ind1],1),0)
        y2 = ndimage.gaussian_filter(np.mean(l[:,stim_ind2],1),0)
        s1 = np.std(l[:,stim_ind1],1)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(l[:,stim_ind2],1)/np.sqrt(np.sum(stim_ind2))
        
        ax3.plot(xaxis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = 'solid')
        ax3.fill_between(xaxis,y1-s1, y1+s1, color = "blue",alpha = 0.5)
        ax3.plot(xaxis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax3.fill_between(xaxis,y2-s2, y2+s2, color = "red",alpha = 0.5)
        
        
        ax2.set_title('unit_'+str(n+1))
        sc = np.mean(score3)
        ax4.set_title(f'{sc:.2f}')
        plt.show()
    
    
    return Xstim, L_on, inter, TT, Y, max_it, score3, init_compare_score, yhat


    
    
# %% Initialize
"""     
Each column of X contains the following information:
    0 : contingency 
    1 : lick vs no lick
    2 : correct vs wrong
    3 : stim 1 vs stim 2
    4 : if exists, would be correct history (previous correct ) 

"""



t_period = 4000
prestim = 4000

window = 50 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 20 # number of cv
ca = 0

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [1]



if ca ==0:
    D_ppc = load_matfile()
    # good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    # good_list = find_good_data_Ca(t_period)
    
    
    
    
# %% Run GLM

Data = {}



for c_ind in c_list:
    # t = 0 
    good_list2 = [];
    for n in np.arange(np.size(D_ppc,0)):
        
        n = int(n)
        if D_ppc[n,4][0][0] > 0:
            try:
                Xstim, L_on, inter, TT, Y, max_it, score3, init_score, yhat  = glm_per_neuron(n,c_ind,1)
                Data[n,c_ind-1] = {"X":Xstim,"coef" : TT, "score" : score3, 'Y' : Y,'init_score' : init_score,
                                    "intercept" : inter,'L' : L_on,"yhat" : yhat}
                good_list2 = np.concatenate((good_list2,[n]))
                print(n)
                
            except KeyboardInterrupt:
                
                break
            except:
            
                print("Break, no fit") 
np.save('SU_AC_0425.npy', Data,allow_pickle= True)     
# Data2 = np.load('RTnew_0411.npy',allow_pickle= True).item()
# test = Data2.item()

# test1 =test(7,2)



# %%


    
    

def extract_onset_times(D_ppc,n):
    # window = 100
    # stim_onset =np.round(D_ppc[n,3][0,D_ppc[n,2][:,0]]*(1e3/window))
    # stim_onset = stim_onset.astype(int)
    # Rt = np.floor(D_ppc[n,6][:,0]*(1e3/window))
    # Rt =Rt.astype(int)
    
    stim_onset = np.round(D_ppc[n,3][:,0]*1e3)
    stim_onset = stim_onset.astype(int)
    if len(stim_onset) > 460:
        stim_onset = np.concatenate((stim_onset[:200],stim_onset[260:460]))
    else:
        stim_onset = np.concatenate((stim_onset[:200],stim_onset[260:])) 
        
    Rt = np.zeros_like(stim_onset)
    Rt[:] = stim_onset[:]+1600
    X = D_ppc[n,2] # task variables
    if np.size(X,0) >400:
        X = X[:400,:]
    
    
    Rt[X[:,2]==0] = 0    
    
    stim_onset = np.floor(stim_onset/window).astype(int)
    Rt = np.floor(Rt/window).astype(int)
    
    if c_ind == 1:
        c1 = stim_onset[1]-100
        c2 = stim_onset[151]
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[1:150]
        r_onset = Rt-c1
        r_onset = r_onset[1:150]

    
    elif c_ind == 3:
        c1 = stim_onset[200]-100
        c2 = stim_onset[D_ppc[n,4][0][0]+26] 
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[200:D_ppc[n,4][0][0]+26]
        r_onset = Rt-c1
        r_onset = r_onset[200:D_ppc[n,4][0][0]+26]



    elif c_ind == 2:       
        c1 = stim_onset[D_ppc[n,4][0][0]+26]-100 
        c2 = stim_onset[399]
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[D_ppc[n,4][0][0]+26:398]
        r_onset = Rt-c1
        # r_onset = r_onset[200:D_ppc[n,4][0][0]+26]
        r_onset = r_onset[D_ppc[n,4][0][0]+26:398]

    return stim_onset2, r_onset

    
for n in np.arange(np.size(good_list2,0)):
    nn = int(good_list2[n])
    # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
    stim_onset,r_onset = extract_onset_times(D_ppc,nn)
    Data[nn,c_ind-1]["stim_onset"] = stim_onset
    Data[nn,c_ind-1]["r_onset"] = r_onset
    
    
# %%

    # %% Normalized population average of task variable weights
# c_ind = 1
d_list = good_list2 > 195
# d_list3 = good_list <= 179
d_list3 = good_list2 <= 195

# Lic = np.where(good_listRu <180)
# Lic = Lic[0][-1]
# good_list_sep = good_listRu[:]

good_list_sep = good_list2[:]

weight_thresh = 5*1e-2


# fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))        
cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
# clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR']
lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted']
ax_sz = len(cmap)


Convdata = {}
pre = 40 # 10 40 
post = 50 # 50 20

pre = int(2000/window)
post= int(6000/window)
xaxis = np.arange(post+pre)- pre
xaxis = xaxis/window

for a in np.arange(ax_sz):
    Convdata[a] = np.zeros((np.size(good_list_sep),pre+post))

for n in np.arange(np.size(good_list_sep,0)):
    nn = int(good_list_sep[n])
    # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
    Model_coef = Data[nn, c_ind-1]["coef"]
    Model_score = Data[nn, c_ind-1]["score"]
    stim_onset2 =  Data[nn, c_ind-1]["stim_onset"]
    stim_onset =  Data[nn, c_ind-1]["stim_onset"]
    
    
    weight = {}
    max_it = [key for key in Model_coef]
    for a in max_it:
        weight[a] = np.zeros((pre+post,np.size(stim_onset))) 
        
            
    for st in np.arange(np.size(stim_onset)):
        for t in max_it:
            if stim_onset[st] <0:
                stim_onset[st] = stim_onset2[st]+15
                
            weight[t][:,st] = np.mean(Model_coef[t][stim_onset[st]-pre: stim_onset[st]+post,:],1)

    for a in max_it:    
        Convdata[a][n,:] = np.mean(weight[a],1) /(np.max(np.abs(np.mean(weight[a],1)))+0.2)
        
fig, axes = plt.subplots(1,1,figsize = (10,8))       
for a in np.arange(ax_sz):
    error = np.std(Convdata[a],0)/np.sqrt(np.size(good_list_sep))
    y = ndimage.gaussian_filter(np.mean(Convdata[a],0),2)
    # y = np.abs(y)
    axes.plot(xaxis,y,c = cmap[a],linestyle = lstyles[a])
    axes.fill_between(xaxis,y-error,y+error,facecolor = cmap[a],alpha = 0.3)
    axes.set_ylim([-0.01,0.50])
    
    
    
# %% histogram of TV encoding
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green']
edgec = cmap
# edgec = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']


# d_list2 = d_list3
def TV_hist():
    good_list_sep = good_list2[:]
    TV = np.empty([1,1])
    for n in np.arange(np.size(good_list_sep,0)):
            nn = int(good_list_sep[n])
            Model_coef = Data[nn, c_ind-1]["coef"]
            score = Data[nn,c_ind-1]["score"]
            max_it = [key for key in Model_coef]
            if np.mean(score) > 0.02:
                TV = np.append(TV, max_it)
    
    TV = TV[1:]
    ax_sz = 8
    B = np.zeros((1,ax_sz))
    for f in np.arange(ax_sz):
        B[0,f] = np.sum(TV == f)
        
    # B = B/np.sum(d_list2)
    B = B/np.size(good_list2)
    fig, axes = plt.subplots(1,1, figsize = (15,5))
    axes.grid(visible=True,axis = 'y')
    axes.bar(np.arange(ax_sz)*3,B[0,:], color = "white", edgecolor = edgec, alpha = 1, width = 0.5, linewidth = 2,hatch = '/')
    axes.set_ylim([0,0.8])
            
TV_hist()

S = np.empty([1,1])

for n in np.arange(np.size(good_list_sep,0)):
    nn = int(good_list_sep[n])
    Model_coef = Data[nn, c_ind-1]["coef"]
    score = Data[nn,c_ind-1]["score"]
    S = np.append(S,np.mean(score))

fig,ax = plt.subplots(1,1, figsize = (5,5))
ax.hist(S,bins = np.arange(0.01,0.8,0.04))




        

# %% plotting weights by peak order
listOv = {}

f = 0
W5 = {}
W5AC= {}
W5IC = {}
max_peak3 ={}
tv_number = {}
b_count = {}
ax_sz = 8
for ind in [0,1]: # 0 is PPCIC, 1 is PPCAC
    b_count[ind] = np.zeros((2,ax_sz))

    for f in np.arange(ax_sz):
        W5[ind,f] = {}

for ind in [0]:
    for f in np.arange(ax_sz):
        list0 = (np.mean(Convdata[f],1) != 0)
        
        # Lg = len(good_list2)
        # Lic = np.where(good_list2 <180)
        # Lic = Lic[0][-1]
        # if ind == 0:
        #     list0[Lic:Lg] = False # PPCIC
        # elif ind == 1:           
        #     list0[0:Lic] = False # PPCAC
        
        # list0ind = np.arange(Lg)
        # list0ind = list0ind[list0]
        list0ind = good_list2[list0]
        W = ndimage.uniform_filter(Convdata[f][list0,:],[0,2], mode = "mirror")
        
        max_peak = np.argmax(np.abs(W),1)
        max_ind = max_peak.argsort()
        
        list1 = []
        list2 = []
        list3 = []
        
        SD = np.std(W[:,:])
        for m in np.arange(np.size(W,0)):
            n = max_ind[m]
            # SD = np.std(W[n,:])
            # if SD< 0.05:
            #     SD = 0.05
            if max_peak[n]> 0:    
                if W[n,max_peak[n]] > 3*SD:
                    list1.append(m)
                    list3.append(m)
                elif W[n,max_peak[n]] <-3*SD:
                    list2.append(m)
                    list3.append(m)
                
        max_ind1 = max_ind[list1]  
        max_ind2 = max_ind[list2]     
        max_ind3 = max_ind[list3]
        max_peak3[ind,f] = max_peak[list3]
        
        listOv[ind,f] = list0ind[list3]
        
        W1 = W[max_ind1]
        W2 = W[max_ind2]    
        W4 = np.abs(W[max_ind3])
        s ='+' + str(np.size(W1,0)) +  '-' + str(np.size(W2,0))
        print(s)
        b_count[ind][0,f] = np.size(W1,0)
        b_count[ind][1,f] = np.size(W2,0)
        W3 = np.concatenate((W1,W2), axis = 0)
        tv_number[ind,f] = [np.size(W1,0),np.size(W2,0)]
        
        W5[ind,f][0] = W3
        W5[ind,f][1] = W2
        if f in [4]: # np.arange(ax_sz): #[4]:
            clim = [-0.7, 0.7]
            fig, axes = plt.subplots(1,1,figsize = (10,10))
            im1 = axes.imshow(W3[:,:], clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
            # im2 = axes[1].imshow(W2, aspect = "auto", interpolation = "None")
            # axes.set_xlim([,40])
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(im1, cax=cbar_ax)
        if ind == 0:
            W5IC[f] = W2
        elif ind == 1:           
            W5AC[f] = W2
        # W4IC = W4
    
# print(np.size(np.intersect1d(listOv[0],listOv[3])))
# np.save('PPC_Hist.npy',listOv,allow_pickle = True)

# np.argmax()

# list0n = good_listRu[list0]
# ind= 1
# np.sum((max_peak3[ind,4] > 54) * (max_peak3[ind,4] < 80 ))

# np.sum((max_peak3[ind,4] > 1) * (max_peak3[ind,4] < 40 ))
# np.sum((max_peak3[ind,4] > 80))

# %% for each timebin, calculate the number of neurons encoding each TV

cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR','Hit_hist','FA_hist']
lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted']

Lic1 = 69
Lg1 = 113

ind = 0 # PPCIC or 1 PPCAC
p = 0 # positive or 1 negative

fig, axes = plt.subplots(1,1,figsize = (10,5))
for f in np.arange(ax_sz):
    list0 = (np.mean(Convdata[f],1) != 0)
        
    # Lg = len(good_list2)
    # Lic = np.where(good_list2 <180)
    # Lic = Lic[0][-1]
    # if ind == 0:
    #     list0[Lic:Lg] = False # PPCIC
    # elif ind == 1:           
    #     list0[0:Lic] = False # PPCAC
        
        # list0ind = np.arange(Lg)
        # list0ind = list0ind[list0]
    list0ind = good_list2[list0]
    W = ndimage.uniform_filter(Convdata[f][list0,:],[0,2], mode = "mirror")
        
    SD = np.std(W[:,:])
    test = np.abs(W5[ind,f][p])>2*SD
    if ind ==0:        
        y = np.sum(test,0)/np.size(list0)
    elif ind == 1:
        y = np.sum(test,0)/Lg1
    if p == 0:
        axes.plot(y,c = cmap[f], linestyle = lstyles[f], linewidth = 3, label = clabels[f] )
        axes.set_ylim([0,0.50])
        axes.legend()
    elif p == 1:
        axes.plot(-y,c = cmap[f], linestyle = lstyles[f], linewidth = 3 )
        axes.set_ylim([-0.30,0])