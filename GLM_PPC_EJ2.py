# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:35:38 2022

@author: Jong Hoon Lee
"""


# import packages 

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
from scipy import stats
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from matplotlib.colors import BoundaryNorm

# from sklearn.metrics import get_scorer_names, d2_tweedie_score, make_scorer

# %%

fname = 'GLM_dataset_220824_new.mat'
np.seterr(divide = 'ignore') 
def load_matfile(dataname = fname):
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_dataset']
    return D_ppc 

# def cross_validation(mdl, X, y, k):
#     ss = ShuffleSplit(n_splits=k, test_size=0.20, random_state=0)
#     theta2 = []
#     score = []
#     for train_index, test_index in ss.split(X):
#         y_train  = y[train_index]
#         X_train = X[train_index]
#         mdl.fit(X_train,y_train)
#         th2 = mdl.coef_
#         sc = mdl.score(X_train,y_train)
#         theta2 = np.concatenate((theta2,th2))
#         score = np.concatenate((score,[sc]))
    
#     score = np.reshape(score, (1,k))
#     theta2 = np.reshape(theta2,(np.size(X,1),k))
    
#     return theta2,score


# %%
def glm_per_neuron(n,t_period,window,k):
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
    Y = [];
    Yhat = [];
    # TT = [];
    TT2 = [];
    Intercept = [];
    CI2 = [];
    score = [];
    # S = np.concatenate((S_pre,S),1)
    # t_period = t_period+prestim
    
    
    # remove conditioning trials 
    
    S = np.concatenate((S[0:200,:],S[D_ppc[n,5][0][0]:,:]),0)
    X = np.concatenate((X[0:200,:],X[D_ppc[n,5][0][0]:,:]),0)
    N_trial2 = np.size(S,0)
    # X = X[:,[0,3]]
    reg = TweedieRegressor(power = 0, alpha = 0)
    # S = ndimage.gaussian_filter(S,sigma = [0,2])

    for w in range(int(t_period/window)):
        y = np.mean(S[:,range(window*w,window*(w+1))],1)*1e3
        X2 = np.column_stack([np.ones_like(y),X])
        ss= ShuffleSplit(n_splits=k, test_size=0.20, random_state=0)
        cv_results = cross_validate(reg, X, y, cv = ss , 
                                    return_estimator = True, 
                                    scoring = 'explained_variance')
        # cv_results2 = cross_val_score(reg, X,y ,cv = 5,scoring = 'r2')
        theta2 = np.zeros((np.size(X,1),k))
        inter2 = np.zeros((1,k))
        pp = 0
        for model in cv_results['estimator']:
            theta2[:,pp] = model.coef_
            inter2[:,pp] = model.intercept_
            pp = pp+1
        theta3 = np.concatenate((np.mean(inter2,1),np.mean(theta2,1)))
        yhat = X2 @ theta3
        
        
        
        score = np.concatenate((score, cv_results['test_score']))
        TT2 = np.concatenate((TT2,np.mean(theta2,1)))
        Intercept = np.concatenate((Intercept,np.mean(inter2,1)))
        CI2 = np.concatenate((CI2,stats.sem(theta2,1)))

        Y = np.concatenate((Y,y))
        Yhat = np.concatenate((Yhat,yhat))
        
    Y = np.reshape(Y,(int(t_period/window),N_trial2)).T
    Yhat = np.reshape(Yhat,(int(t_period/window),N_trial2)).T
    # TT = np.reshape(TT,(int(t_period/window),np.size(X,1))).T
    
    # TT = np.reshape(TT,(np.size(X,1),int(t_period/window)))
    
    TT2 = np.reshape(TT2,(int(t_period/window),np.size(X,1))).T
    CI2 = np.reshape(CI2,(int(t_period/window),np.size(X,1))).T
    score = np.reshape(score,(int(t_period/window),k))
    
    
    
    
    
    
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))
    
    
    # fig, ax = plt.subplots()
    cmap = ['tab:purple', 'tab:orange', 'tab:green','tab:blue']
    for c in range(np.size(X,1)):        
        # ax1.plot(ndimage.gaussian_filter(TT[c,:],3),linewidth = 2.0, color = cmap[c])
        ax2.plot(ndimage.gaussian_filter(TT2[c,:],3),linewidth = 2.0, color = cmap[c])
        x_axis = np.linspace(1, 90, 90);
        ax2.fill_between(x_axis,(ndimage.gaussian_filter(TT2[c,:],3) - CI2[c,:]),
                        (ndimage.gaussian_filter(TT2[c,:],3 )+ CI2[c,:]), color=cmap[c], alpha = 0.2)
    
    # ax1.legend(["stim","action","correct"])
    ax2.legend(["contin","action","correct","stim"])

    e_lines = np.array([0,500,500+int(D_ppc[n,3]),2500+int(D_ppc[n,3])])/window
    # e_lines = e_lines+prestim/window
    # ax1.vlines(x =e_lines, 
    #           ymin = np.amin(ndimage.gaussian_filter(TT,sigma = [0,3])), 
    #           ymax = np.amax(ndimage.gaussian_filter(TT,sigma = [0,3])),
    #           linestyles = 'dashed',
    #           colors = 'black', 
    #           linewidth = 2.0)
    
    ax2.vlines(x =e_lines, 
              ymin = np.amin(ndimage.gaussian_filter(TT2,sigma = [0,3])), 
              ymax = np.amax(ndimage.gaussian_filter(TT2,sigma = [0,3])),
              linestyles = 'dashed',
              colors = 'black', 
              linewidth = 2.0)
    
    ax4.plot(np.mean(score,1)*1e2)
    
    
    # plot rasterplot for different conditions (stim)
    # norm = BoundaryNorm([-1, -.2, .2, 1], 256)
    stim_ind = X[:,1] == 1
    
    # imx = ax1.pcolormesh(Y[stim_ind,:], cmap="gray_r")
    # imx2 = ax3.pcolormesh(Y[np.invert(stim_ind),:],cmap = "gray_r")

    ax1.plot(ndimage.gaussian_filter(np.mean(Y[stim_ind,:],0),2),linewidth = 2.0, color = cmap[2])
    ax1.plot(ndimage.gaussian_filter(np.mean(Y[np.invert(stim_ind),:],0),2),linewidth = 2.0, color = cmap[3])
    
    ax3.plot(ndimage.gaussian_filter(np.mean(Yhat[stim_ind,:],0),2),linewidth = 2.0, color = cmap[2])
    ax3.plot(ndimage.gaussian_filter(np.mean(Yhat[np.invert(stim_ind),:],0),2),linewidth = 2.0, color = cmap[3]) 
    
    plt.title('unit_'+str(n))
    plt.show()
    Model_Theta = TT2
    
    return X, Y, Yhat, Model_Theta
        

# n = 0 # neuron id, will do this for all neurons
t_period = 4500
window = 50 # window of 100ms, averaging firing rates to this window 
k = 10 # number of cv
# n = 88
for n in range(170):
    X, Y, Yhat, Model_Theta = glm_per_neuron(n, t_period, window,k)
    
    






























    