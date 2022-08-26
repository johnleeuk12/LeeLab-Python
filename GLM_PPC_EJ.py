# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 09:09:11 2022

GLM analysis for rule-switching SU data from PPC, recorded by EJ


@author: Jong Hoon Lee
"""

# import packages 

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
from sklearn.linear_model import TweedieRegressor


# %%

fname = 'GLM_dataset_220824_new.mat'

def load_matfile(dataname = fname):
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_dataset']
    return D_ppc 

# %%

def glm_per_neuron(n,t_period,window):
    D_ppc = load_matfile()
    S_all = np.zeros((1,max(D_ppc[n,2][:,0])+t_period+100))
    # S_all = np.zeros((1,max(D_ppc[n,0][:,0])))
    N_trial = np.size(D_ppc[n,2],0)
    prestim = 500
    
    
    for sp in np.array(D_ppc[n,0]):
        if sp < np.size(S_all,1):
            S_all[0,sp[0]-1] = 1  #spike time starts at 1 but indexing starts at 0
                
    
    S = np.zeros((N_trial,t_period))
    S_pre = np.zeros((N_trial,prestim))
    for tr in range(N_trial):
        S[tr,:] = S_all[0,D_ppc[n,2][tr,0]-1:D_ppc[n,2][tr,0]+t_period-1]
        S_pre[tr,:] = S_all[0,D_ppc[n,2][tr,0]-prestim-1:D_ppc[n,2][tr,0]-1]
    
    X = D_ppc[n,2][:,2:6]
    # X = D_ppc[n,2][:,5]
    Y = [];
    TT = [];
    
    Yhat = [];
    score = [];
    # S = np.concatenate((S_pre,S),1)
    # t_period = t_period+prestim
    
    
    # remove conditioning trials 
    
    S = np.concatenate((S[0:200,:],S[D_ppc[n,5][0][0]:,:]),0)
    X = np.concatenate((X[0:200,:],X[D_ppc[n,5][0][0]:,:]),0)
    N_trial2 = np.size(S,0)
    # S = ndimage.gaussian_filter(S,sigma = [0,2])
    
    
    reg = TweedieRegressor(power = 0, alpha = 1)
    # X = X[:,3]
    for w in range(int(t_period/window)):
        y = np.mean(S[:,window*w:window*(w+1)],1)*1e3
        # y = ndimage.gaussian_filter(y,2)
        X2 = np.column_stack([np.ones_like(y),X])
        theta = np.linalg.inv(X2.T @ X2) @ X2.T @ y
        theta2 = theta
        theta = theta[1:]
        
        # reg.fit(X,y)
        # theta = reg.coef_
        # theta2 = np.concatenate(([reg.intercept_], reg.coef_))
        # sc = reg.score(X,y)
        Y = np.concatenate((Y,y))
        TT = np.concatenate((TT,theta))
        # score = np.concatenate((score,[sc]))
        yhat = X2 @ theta2
        Yhat = np.concatenate((Yhat, yhat))
        
    # Y = np.reshape(Y,(N_trial2, int(t_period/window)))
    Y = np.reshape(Y,(int(t_period/window),N_trial2)).T
    Yhat = np.reshape(Yhat,(int(t_period/window),N_trial2)).T
    TT = np.reshape(TT,(int(t_period/window),np.size(X,1))).T
    
    fig, ((ax,ax4),(ax1,ax2)) = plt.subplots(2,2)
    cmap = ['tab:purple', 'tab:orange', 'tab:green','tab:blue']

    for c in range(np.size(X,1)):
        
        ax.plot(ndimage.gaussian_filter(TT[c,:],3),linewidth = 2.0,color = cmap[c])
    # ax.plot(ndimage.gaussian_filter(TT,3),linewidth = 2.0)

    ax.legend(["contin","action","correct","stim"])
    
    e_lines = np.array([0,500,500+int(D_ppc[n,3]),2500+int(D_ppc[n,3])])/window
    # e_lines = e_lines+prestim/window
    ax.vlines(x =e_lines, 
              ymin = np.amin(ndimage.gaussian_filter(TT,sigma = [0,3])), 
              ymax = np.amax(ndimage.gaussian_filter(TT,sigma = [0,3])),
              linestyles = 'dashed',
              colors = 'black', 
              linewidth = 2.0)
    # plt.title('unit_'+str(n))
    # plt.show()



    # fig, (ax1,ax2) = plt.subplots(2,1)
    #plot y, for diff stim 
    stim_ind = X[:,2] == 1
    ax1.plot(ndimage.gaussian_filter(np.mean(Y[stim_ind,:],0),2),linewidth = 2.0, color = cmap[2])
    ax1.plot(ndimage.gaussian_filter(np.mean(Y[np.invert(stim_ind),:],0),2),linewidth = 2.0, color = cmap[3])
    
    ax2.plot(ndimage.gaussian_filter(np.mean(Yhat[stim_ind,:],0),2),linewidth = 2.0, color = cmap[2])
    ax2.plot(ndimage.gaussian_filter(np.mean(Yhat[np.invert(stim_ind),:],0),2),linewidth = 2.0, color = cmap[3])    
    
    # fig, ax = plt.subplots()
    ax4.plot(score,linewidth = 2.0)
    # ax4.pcolormesh(S, cmap="gray_r")
    plt.title('unit_'+str(n))
    plt.show()
    return X, Y, Yhat
        

# n = 0 # neuron id, will do this for all neurons
t_period = 4500
window = 200 # window of 100ms, averaging firing rates to this window 


# for n in range(10): 
    
n = 88 
X, Y, Yhat = glm_per_neuron(n, t_period, window)
    
    






























    