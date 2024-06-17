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


fname = 'CaData_all_all_session_v2_corrected.mat'
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

# %% 

trmax = 8
# D2 = {}
# for tr in np.arange(trmax+2):
#     for ind in [0,1]:
#         D2[ind,tr] = D[ind,tr]- np.mean(D[ind,tr][:,10:30])
fig, axes = plt.subplots(1,1, figsize = (10,10))
for tr in [8,4,9]:
    axes.plot(np.mean(D[1,tr],0))


# %%
from scipy import linalg

pca = {}
max_k = 20;
# d_list = np.logical_and(good_list > 179, good_list < 600)
d_list = good_list > 179

d_list3 = good_list <= 179

trmax = 8

d_list2 = d_list
# d_list2 = good_list>-1
# fig, axs = plt.subplots(trmax+2,6,figsize = (20,30))

sm = 0
R = {}
for g in  np.arange(trmax+2):
    for ind in [0,1]:
        pca[ind,g] = PCA(n_components=64)
        # R[g] = np.concatenate((ndimage.gaussian_filter(D[1,g][d_list2,:].T,[sm,0]),ndimage.gaussian_filter(D[0,g][d_list2,:].T,[sm,0])),0)
        # R[g] = ndimage.gaussian_filter(D[1,g][d_list2,:].T,[sm,0]) + ndimage.gaussian_filter(D[1,g][d_list2,:].T,[sm,0])
        R[g] = D[ind,g][d_list2,:].T
        # R[g] = R[g]/2
        # R[g] = np.concatenate((D[0,g][d_list2,:].T,D[1,g][d_list2,:].T),1)
    # 
        # R[g] =ndimage.gaussian_filter(D[1,g][d_list2,:].T,[sm,0])
        # R = ndimage.gaussian_filter(D_r[g][:,:].T,[sm,0])
        # R = ndimage.gaussian_filter(zscore(D[1,g][d_list2,:], axis = 1),[sm,0])
        test = pca[ind,g].fit_transform(ndimage.gaussian_filter(R[g],[1,0]))        



U = {}

for g1 in  np.arange(trmax+2):
    for ind1 in [0,1]:
        Rt = D[ind1,g1][d_list2,:].T
        U[ind1,g1], s, Vh = linalg.svd(Rt.T)
      


        
        
# %%


g1 = 9 # space of comparison (projection subspace)\
ind1 = 1
g2 = 0 # projection data
ind2 = 1
    
k = 10

V1 = np.zeros((1,160))
V = np.zeros((1,160))    
for t in np.arange(160):
        # Rt = D[ind1,g1][d_list2,10*t:10*(t+1)-1].T
        Rt = D[ind1,g1][d_list2,t].T
        # V1[0,t] = 1-np.linalg.norm(Rt  - np.dot(np.dot(Rt ,pca[ind1,g1].components_[:k,:].T),
        #                                                                 pca[ind1,g1].components_[:k,:]))/np.linalg.norm(Rt)
        V1[0,t] = 1-np.linalg.norm(Rt  - np.dot(np.dot(Rt ,U[ind1,g1][:,:k]),U[ind1,g1][:,:k].T))/np.linalg.norm(Rt)
        
for t in np.arange(160):
        # Rt = D[ind1,g1][d_list2,10*t:10*(t+1)-1].T
        Rt = D[ind1,g1][d_list2,t].T
        # V2 = 1-np.linalg.norm(Rt - np.dot(np.dot(Rt,pca[ind2,g2].components_[:k,:].T),
        #                                                                 pca[ind2,g2].components_[:k,:]))/np.linalg.norm(Rt )
        V2 = 1-np.linalg.norm(Rt  - np.dot(np.dot(Rt ,U[ind2,g2][:,:k]),U[ind2,g2][:,:k].T))/np.linalg.norm(Rt)
        V[0,t] = V2*V1[0,t] # /np.max(V1)
fig, ax = plt.subplots(1,1, figsize = (10,10))
# ax.plot(V1[0,:])
ax.plot(V[0,:])        

# %%
Rt = D[ind1,g1][d_list2,:].T
test =np.dot(np.dot(Rt,U[ind1,g1][:,:k]),U[ind1,g1][:,:k].T)

fig ,axes = plt.subplots(1,1, figsize = (10,10)) 
axes.plot(np.mean(test,1))   
axes.plot(np.mean(R[g1],1))   


def var_exp(g1,ind1,g2,ind2,k):
    V = np.zeros((1,160))
    

    
    V1 = np.zeros((1,160))
    
    for t in np.arange(160):
        # Rt = D[ind1,g1][d_list2,10*t:10*(t+1)-1].T
        Rt = D[ind1,g1][d_list2,t].T
        # V1[0,t] = 1-np.linalg.norm(Rt  - np.dot(np.dot(Rt ,pca[ind1,g1].components_[:k,:].T),
        #                                                                 pca[ind1,g1].components_[:k,:]))/np.linalg.norm(Rt)
        V1[0,t] = 1-np.linalg.norm(Rt  - np.dot(np.dot(Rt ,U[ind1,g1][:,:k]),U[ind1,g1][:,:k].T))/np.linalg.norm(Rt)
    # fig, ax = plt.subplots(1,1, figsize = (10,10))
    # ax.plot(V1[0,:])    
        
    for t in np.arange(160):
        # Rt = D[ind1,g1][d_list2,10*t:10*(t+1)-1].T
        Rt = D[ind1,g1][d_list2,t].T
        # V2 = 1-np.linalg.norm(Rt - np.dot(np.dot(Rt,pca[ind2,g2].components_[:k,:].T),
        #                                                                 pca[ind2,g2].components_[:k,:]))/np.linalg.norm(Rt )
        V2 = 1-np.linalg.norm(Rt  - np.dot(np.dot(Rt ,U[ind2,g2][:,:k]),U[ind2,g2][:,:k].T))/np.linalg.norm(Rt)
        V[0,t] = V2 # /np.max(V1)

    return V



g1 = 8 # space of comparison (projection subspace)\
ind1 = 1
g2 = 0 # projection data
ind2 = 1
    
k = 20

fig, ax = plt.subplots(1,1, figsize = (10,10))
             
for g2 in [0,1,2]:
    V = var_exp(g1,ind1,g2,ind2,k)
    # ax.plot(V1[0,:])
    ax.plot(V[0,:])   
    

# ax.plot(V[0,:]*np.max(V1))


# %%
cmap = ["tab:blue","tab:purple","tab:green"]
lstyles = ["dotted","solid"]

fig, ax = plt.subplots(1,1, figsize = (10,10))
g1 =9
ind1 = 1
for tr in [0,1,2]:
    for ind in [0,1]:
        if tr == 0:
            V =var_exp(g1,ind1,8,ind,k)
        elif tr == 1:
            V =var_exp(g1,ind1,5,ind,k)
        elif tr == 2:
            V =var_exp(g1,ind1,9,ind,k)
        
        V = V-np.mean(V[0,0:20])
        ax.plot(V[0,:], color = cmap[tr],linestyle = lstyles[ind])
        ax.set_ylim([-0.1,1])











    