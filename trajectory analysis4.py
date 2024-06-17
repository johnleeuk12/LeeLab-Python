# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:43:09 2024


        
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


# %% File name and directory

# change fname for filename
# fname = 'CaData_all_withlicktime_correctedv2.mat'


# fname = 'CaData_all_all_session_v2_corrected.mat'
# fname = 'CaData_PIC_2.mat'
fname = 'CaData_all_session_v3_corrected.mat'


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
def makeD(lenx):
    D = {}
    trmax = 8
    alpha = 0.2
    for tr in np.arange(trmax):
        D[0,tr] = np.zeros((len(good_list),lenx))
        D[1,tr] = np.zeros((len(good_list),lenx))
    
    
    
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
                trial_ind = np.argwhere(X[:,0] == ind)
                np.random.shuffle(trial_ind)
                if np.size(trial_ind) >5:
                # D[ind,tr][m,:] = np.mean(Y[X[:,0] == ind,:],0)
                    Y2 = np.mean(Y[trial_ind[5:,0],:],0)-np.mean(Y[:,0:10])
                    # Y2 = np.mean(Y[trial_ind[5:,0],:],0)
                    # D[ind,tr][m,:] = np.mean(Y[trial_ind[5:,0],:],0)-np.mean(Y[:,0:10])
                else:
                    Y2 = np.mean(Y[trial_ind[:,0],:],0)-np.mean(Y[:,0:10])
                    # Y2 = np.mean(Y[trial_ind[:,0],:],0)
                    # D[ind,tr][m,:] = np.mean(Y[trial_ind[:,0],:],0)-np.mean(Y[:,10:30])
                Y3 = Y2/(np.max(np.mean(Y,0)-np.mean(Y[:,0:10])) + alpha)
                # Y3 = Y2/(np.max(np.mean(Y,0)) + alpha)
                
                
                # D[ind,tr][m,:] = D[ind,tr][m,:]/(np.max(np.mean(Y,0)-np.mean(Y[:,10:30])) + alpha) # for original trajectories with Go+ NG
                D[ind,tr][m,:] = Y3
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
                trial_ind = np.argwhere(X[:,0] == ind)
                np.random.shuffle(trial_ind)
                D[ind,trmax+c_ind-1][m,:] = np.mean(Y[trial_ind[5:,0],:],0)-np.mean(Y[:,0:10])
                # D[ind,trmax+c_ind-1][m,:] = np.mean(Y[X[:,0] == ind,:],0)
                D[ind,trmax+c_ind-1][m,:] = D[ind,trmax+c_ind-1][m,:]/(np.max(np.mean(Y,0)-np.mean(Y[:,0:10])) + alpha)
                # D[ind,trmax+c_ind-1][m,:] = D[ind,trmax+c_ind-1][m,:]/((np.max(Y)-np.min(Y)) + alpha)
            # 
            # D[2,trmax+c_ind-1][m,:] = (np.mean(Y,0)-np.mean(Y[:,10:30]))/(np.max(np.mean(Y,0)) + alpha)
    
            m += 1
    return D                    

c_ind =3
D_cv = {}
for cv in np.arange(2):
    print('cv',cv)
    D_cv[cv] = makeD(lenx)
    
    
# %% draw Go and No-Go

# fig, axes = plt.subplots(1,1, figsize = (10,10))
# axes.plot(Y3)

d_list = good_list > 179
d_list3 = good_list <= 179

d_list2 = d_list3

# trmax = 8
# for tr in np.arange(trmax+2):
#     for ind in [0,1]:
#         D_cv[cv][ind,tr] = D_cv[cv][ind,tr]- np.mean(D_cv[cv][ind,tr][:,10:30])


fig, axes = plt.subplots(1,1, figsize = (10,10))
for tr in [8,4,9]:
    axes.plot(np.mean(D_cv[1][1,tr][d_list3,:],0))

# %% 
def make_PCA_weight(D,d_list):
    W = {}
    R = {}
    c_list = [0,1,2]
    for c_ind in c_list:
        R[c_ind]= np.empty( shape=(0,np.sum(d_list)) )
    
    sm = 0
    for c_ind in [0,1,2]:
        if c_ind == 0:
            tr_list = [0,1,2]
        elif c_ind == 1: 
            tr_list = [4]
        elif c_ind == 2:
            tr_list = [5,6,7]
        for tr in tr_list:
            for ind in [0,1]:
                R[c_ind] = np.concatenate((R[c_ind],ndimage.gaussian_filter(D[ind,tr][d_list,:].T,[sm,0])),0)
    
        pca = PCA(n_components=64)
       
        test = {}
    for c_ind in c_list:
        test[c_ind] = pca.fit_transform(R[c_ind])
        W[c_ind] = pca.components_.T        
    return W


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
                    
            #if tr == g and ind == 1:
            ax.auto_scale_xyz(x,y,z)
                # ax.set_xlim([-2,5])
                # ax.set_ylim([-3,3])
                # ax.set_zlim([-2,2])
                
                

            
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
d_list = good_list > 179
d_list3 = good_list <= 179

d_list2 = d_list
sm = 0
c_ind = 1


traj = {}
traj[0] = {}
traj[1] = {}
# trmax = 7
# m = 0 
sm = 0

distance = {}
distance[0] = np.zeros((20,160))
distance[1] = np.zeros((20,160))
distance[2] = np.zeros((20,160))
distance[3] = np.zeros((20,160))
distance[4] = np.zeros((20,160))
distance[5] = np.zeros((20,160))

for cv in np.arange(1):
    W = make_PCA_weight(D_cv[cv], d_list2)
    # P = {}
    # P = []
    P = np.empty(shape=(0,160))
    trmax = 8
    sm = 5
    for tr in np.arange(trmax):
        if tr != 3:
            for ind in [0,1]:
                test = np.dot(ndimage.gaussian_filter(D_cv[cv][ind,tr][d_list2,:].T,[sm,0]),W[c_ind])
                if P.size ==0:
                    P = test.T
                else:                   
                    P =  np.concatenate((P,test.T),1)
    test= P.T
    m = 0 
    trmax = 7
    for tr in np.arange(trmax):
        for ind in [0,1]:
            # traj[ind][tr] = ndimage.gaussian_filter(test[m*160:(m+1)*160,:],[sm,0])
            traj[ind][tr] = test[m*160:(m+1)*160,:]
            traj[ind][tr] = traj[ind][tr]- np.mean(traj[ind][tr][10:30,:],0)
            m += 1
    
    for ind in [0,1]:
        traj[ind][trmax] = (traj[ind][0] + traj[ind][1] + traj[ind][2])/3
        traj[ind][trmax+1] = (traj[ind][4] + traj[ind][5] + traj[ind][6])/3
        
    distance[0][cv,:] =np.linalg.norm((traj[1][trmax][:,:3] -traj[1][3][:,:3]),axis = 1)
    distance[1][cv,:] =np.linalg.norm((traj[1][trmax][:,:3] -traj[1][trmax+1][:,:3]),axis = 1)
    distance[2][cv,:] =np.linalg.norm((traj[1][3][:,:3] -traj[1][trmax+1][:,:3]),axis = 1)
    
    distance[3][cv,:] =np.linalg.norm((traj[1][trmax][:,:3] -traj[0][trmax][:,:3]),axis = 1)
    distance[4][cv,:] =np.linalg.norm((traj[1][3][:,:3] -traj[0][3][:,:3]),axis = 1)
    distance[5][cv,:] =np.linalg.norm((traj[1][trmax+1][:,:3] -traj[0][trmax+1][:,:3]),axis = 1)



# fig, ax = plt.subplots(1,1, figsize = (20,10))
# xaxis = np.arange(160)-20
# for ind in [3,4,5]:
#     ax.plot(xaxis, np.mean(distance[ind],0))
    # ax.fill_between(xaxis,np.mean(distance[ind],0)+np.std(distance[ind],0),np.mean(distance[ind],0)-np.std(distance[ind],0), alpha = 0.3)



trmax = 7
draw_traj4(traj,0,trmax,0,trmax)


# %%

def make_PCA_weight2(D,d_list):
    W = {}
    # R = {}
    # c_list = [0,1,2]
    # for c_ind in c_list:
    #     R[c_ind]= np.empty( shape=(0,np.sum(d_list)) )
    R= np.empty( shape=(0,np.sum(d_list)) )
    D2 = {}
    # sm = 0
    for c_ind in [0,1,2]:
        for ind in [0,1]:
            D2[ind] = []
            if c_ind == 0:
                # D2[ind]= D[ind,0][d_list,:].T + D[ind,1][d_list,:].T +D[ind,2][d_list,:].T
                # D2[ind] = D2[ind]/3
                D2[ind] = D[ind,8][d_list,:].T
            elif c_ind ==1:
                D2[ind] = D[ind,4][d_list,:].T
            elif c_ind == 2:
                # D2[ind]= D[ind,5][d_list,:].T + D[ind,6][d_list,:].T +D[ind,7][d_list,:].T
                # D2[ind] = D2[ind]/3     
                D2[ind] = D[ind,9][d_list,:].T
            R = np.concatenate((R,D2[ind]))        
    pca = PCA(n_components=64)
    test = pca.fit_transform(R)
    W = pca.components_.T        
    return R,W,test,pca

d_list2 = d_list
cv = 0
R,W,test,pca  = make_PCA_weight2(D_cv[cv], d_list2)

fig, axs = plt.subplots(1,3,figsize = (20,10))
m = 0
for t in np.arange(3):
    axs[t].plot(np.mean(R[t*320+160:t*320+320,:],1), linewidth = 4)



fig, axs = plt.subplots(1,6,figsize = (20,10))

test = test.T
for t in range(0,5):
    axs[t].plot(test[t,:], linewidth = 4)
# axs[5].plot(np.cumsum(pca[g].explained_variance_ratio_), linewidth = 4)

# %%
dist = {}
for ind in [0,1,2]:
    dist[ind] = np.zeros((20,160))
d_list2 = d_list3


traj = {}
traj[0] = {}
traj[1] = {}
# trmax = 7
# m = 0 
sm = 0

for cv in np.arange(1):

    R,W,test,pca  = make_PCA_weight2(D_cv[cv], d_list2)
    P = np.empty(shape=(0,160))
    
    trmax = 8
    for tr in np.arange(trmax):
        if tr != 3:
            for ind in [0,1]:
                test = np.dot(ndimage.gaussian_filter(D_cv[cv][ind,tr][d_list2,:].T,[0,0]),W)
                if P.size ==0:
                    P = test.T
                else:                   
                    P =  np.concatenate((P,test.T),1)
    test= P.T
    
    
    m = 0 
    sm =8
    trmax = 7
    # make trajectories
    for tr in np.arange(trmax):
        for ind in [0,1]:
            traj[ind][tr] = ndimage.gaussian_filter(test[m*160:(m+1)*160,:],[sm,0])
            # traj[ind][tr] = test[m*160:(m+1)*160,:]
            traj[ind][tr] = traj[ind][tr]- np.mean(traj[ind][tr][10:30,:],0)
            m += 1
        
    for ind in [0,1]:
        traj[ind][trmax] = (traj[ind][0] + traj[ind][1] + traj[ind][2])/3
        traj[ind][trmax+1] = (traj[ind][4] + traj[ind][5] + traj[ind][6])/3
    
    # calculate distance between trajectories
    
    k = 10
    # dist[cv,1] = np.linalg.norm((traj[1][trmax][:,:k] -traj[1][trmax+1][:,:k]),axis = 1)
    # dist[cv,2] = np.linalg.norm((traj[1][3][:,:k] -traj[1][trmax+1][:,:k]),axis = 1)
    # dist[cv,0] = np.linalg.norm((traj[1][trmax][:,:k] -traj[1][3][:,:k]),axis = 1)

    dist[1][cv,:] = np.linalg.norm((traj[1][trmax][:,:k] -traj[0][trmax][:,:k]),axis = 1)
    dist[2][cv,:] = np.linalg.norm((traj[1][trmax+1][:,:k] -traj[0][trmax+1][:,:k]),axis = 1)
    dist[0][cv,:] = np.linalg.norm((traj[1][3][:,:k] -traj[0][3][:,:k]),axis = 1)

xaxis = np.arange(160)-20
fig, axs = plt.subplots(1,1,figsize = (15,10))
for ind in [0,1,2]:
    ymax = np.max([np.max(dist[0]),np.max(dist[1]),np.max(dist[2])])
    y = np.mean(dist[ind],0)/ymax
    sd = np.std(dist[ind],0)/(ymax*np.sqrt(np.size(dist[ind],0)))
    axs.plot(xaxis, y, linewidth = 2)
    # axs.fill_between(xaxis,y+sd, y-sd, alpha = 0.5)



# %% make figure with last cv
fig, axs = plt.subplots(2,6,figsize = (30,10))
clabels = ["0-50","50-100","100-150","RT","0-50","FA",'Miss','CR','Hit_hist','FA_hist']

for ind in [0,1]:
    if ind == 0:
        colors = cm.autumn(np.linspace(0,1,trmax))
    elif ind == 1:
        colors = cm.winter(np.linspace(0,1,trmax))
    for t in range(0,5):
        for tr in np.arange(trmax):
            if tr !=3:
                axs[0,t].plot(traj[ind][tr][:,t], linewidth = 2,color = colors[tr],label = "0")
            if tr == 3:
                if ind ==1:
                    axs[0,t].plot(traj[ind][tr][:,t], linewidth = 2,color = "tab:purple", linestyle = 'solid',label = "0")
                elif ind == 0:
                    axs[0,t].plot(traj[ind][tr][:,t], linewidth = 2,color = "tab:purple", linestyle = 'dotted',label = "0")
    for t in range(5,10):
        for tr in np.arange(trmax):
            if tr !=3:
                axs[1,t-5].plot(traj[ind][tr][:,t], linewidth = 2,color = colors[tr],label = "0")
            if tr == 3:
                if ind ==1:
                    axs[1,t-5].plot(traj[ind][tr][:,t], linewidth = 2,color = "tab:purple", linestyle = 'solid',label = "0")
                elif ind == 0:
                    axs[1,t-5].plot(traj[ind][tr][:,t], linewidth = 2,color = "tab:purple", linestyle = 'dotted',label = "0")

for ind in [0,1]:
    for t in range(0,5):
        axs[ind,t].legend()

trmax = 7
# axs[1,5].plot(np.cumsum(pca.explained_variance_ratio_), linewidth = 4)
# draw_traj4(traj,0,trmax,0,trmax)
# %% calculating distance between trajectories, using 5 first PCs 

dist = {}
k = 5
# dist[1] = np.linalg.norm((traj[1][trmax][:,:k] -traj[1][trmax+1][:,:k]),axis = 1)
# dist[2] = np.linalg.norm((traj[1][3][:,:k] -traj[1][trmax+1][:,:k]),axis = 1)
# dist[0] = np.linalg.norm((traj[1][trmax][:,:k] -traj[1][3][:,:k]),axis = 1)

dist[1] = np.linalg.norm((traj[1][trmax][:,:k] -traj[0][trmax][:,:k]),axis = 1)
dist[2] = np.linalg.norm((traj[1][trmax+1][:,:k] -traj[0][trmax+1][:,:k]),axis = 1)
dist[0] = np.linalg.norm((traj[1][3][:,:k] -traj[0][3][:,:k]),axis = 1)





# %% Calculate distance between point A and point B of trajectories

TL = {}
for ind in [0,1]:
    TL[ind] = np.zeros((1,trmax))
    for tr in np.arange(trmax):
        # test = np.linalg.norm((traj[ind][tr][1:,:2]-traj[ind][tr][:159,:2]),axis = 1)
        # TL[ind][0,tr] = np.sum(test[20:140])
        TL[ind][0,tr] = np.linalg.norm((traj[ind][tr][20,:3]-traj[ind][tr][110,:3]))
    
fig, axes = plt.subplots(1,2,figsize = (20,10))
for ind in [0,1]:
    axes[ind].bar(np.arange(trmax),TL[ind][0,:]/np.max(TL[1]))
# axes[ind].errorbar(np.arange(4),mean_TD[0,:],std_TD[0,:], ecolor = 'black', elinewidth = 2, barsabove = True,linewidth = 0)
    axes[ind].set_ylim([0.,1.1])


# %% calculate trajectory length

TL = {}
for ind in [0,1]:
    TL[ind] = np.zeros((1,trmax))
    for tr in np.arange(trmax):
        test = np.linalg.norm((traj[ind][tr][1:,:2]-traj[ind][tr][:159,:2]),axis = 1)
        TL[ind][0,tr] = np.sum(test[20:140])
        # TL[ind][0,tr] = np.linalg.norm((traj[ind][tr][20,:3]-traj[ind][tr][30,:3]))
    
fig, axes = plt.subplots(1,2,figsize = (20,10))
for ind in [0,1]:
    axes[ind].bar(np.arange(trmax),TL[ind][0,:]/np.max(TL[1]))
# axes[ind].errorbar(np.arange(4),mean_TD[0,:],std_TD[0,:], ecolor = 'black', elinewidth = 2, barsabove = True,linewidth = 0)
    axes[ind].set_ylim([0.,1.1])