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
from sklearn.linear_model import TweedieRegressor, Ridge, ElasticNet, Lasso, LogisticRegression
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
    # Y = np.zeros((N_trial,int((2*(t_period+prestim))/window)))
    # Y[0,:] = np.concatenate((Yraw[0,D_ppc[n,2][0,0]-1 - int(prestim/window): D_ppc[n,2][0,0] + int(t_period/window)-1],
    #                         Yraw[0,D_ppc[n,2][0,0]-1 - int(prestim/window): D_ppc[n,2][0,0] + int(t_period/window)-1]))
    # for tr in range(1,N_trial):
    #     Y[tr,:] = np.concatenate((Yraw[0,D_ppc[n,2][tr-1,0]-1 - int(prestim/window): D_ppc[n,2][tr-1,0] + int(t_period/window)-1],
    #                               Yraw[0,D_ppc[n,2][tr,0]-1 - int(prestim/window): D_ppc[n,2][tr,0] + int(t_period/window)-1]))

    
    
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
        L2 = L2[D_ppc[n,4][0][0]:,:]
        

    
    # Add reward  history
    Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
    Xpre = Xpre[:,None]
    
    # 10/05 adding stim history, inverting sign for 10kHz = Go
    Xprestim = np.concatenate(([0],-1*X[0:-1,3]+1),0)
    Xprestim = Xprestim[:,None]
    # Add reward instead of action
    # currently it's Contingency, Stimulus, Reward, and Reward History
    # X2 = np.column_stack([X[:,0],X[:,3],
    #                       X[:,2]*X[:,1],Xpre]) 

    X2 = np.column_stack([Xprestim,-1*X[:,3]+1,
                          X[:,2]*X[:,1],Xpre]) 


    
    return X2,Y, L2, Rt

# %% glm_per_neuron function code

def glm_per_neuron(n,t_period,prestim,window,k,c_ind,ca,fig_on): 
    # if using spike data
    if ca == 0:
        X, Y, Y2,L = import_data_w_spikes(n,prestim,t_period,window,c_ind)
    else:
    # if using Ca data
        X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
        Y2 = Y
    
    
    # t_period = (t_period+prestim)*2
    t_period = t_period + prestim
    Yhat = [];
    Yhat1 = [];
    Yhat2 = [];
    TT2 = [];
    Intercept = [];
    CI2 = [];
    score = [];
    N_trial2 = np.size(X,0)

    
    # reg = TweedieRegressor(power = 0, alpha = 0)
    reg = ElasticNet(alpha = 4*1e-2, l1_ratio = 0.5) #Using a linear regression model with Ridge regression regulator set with alpha = 1
    # reg = Ridge(alpha = 4*1e-2)
    for w in range(int(t_period/window)):
        y = Y2[:,w]
        # l = L[:,w]*0
        # X2 = np.column_stack([np.ones_like(y),X[:,0],l,X[:,2:]])
        # X = np.column_stack([X[:,0],l,X[:,2:]])
        # X3 = np.column_stack([l,X])
        X3 = np.column_stack([X])
        # if c_ind == 1 or c_ind == 2 or c_ind ==3:
        #     X3[:,0] = 0
            
        # Xm = np.zeros_like(X3)
        # Xm[:,m_ind] = 1
        # X3 = X3*Xm
        # adding kernels to each task variable
        if w*window <= prestim-2*window:
            X3[:,1:3] = 0;
            X3[:,2] = 0;
            X3[:,1] = 0;
        elif w*window <= prestim+1500-2*window:
            X3[:,2]= 0;
        elif w*window > prestim+1500-2*window:
            X3[:,1] = 0;
            # if ca == 0:
            #     X3[:,2]= 0;
            # elif ca == 1:
            #     for tr in np.arange(np.size(L,0)):
            #         if np.isnan(Rt[tr,0]):
            #             X3[tr,2] = 0;
            #         else:
            #             if w*window <= prestim + Rt[tr,0]*1e3 -window:
            #                 X3[tr,2] = 0;
                        
        

        
        
        
        # ==== Initial run ====
        score2 = {}
        Nvar = np.size(X,1)
        mi_score = np.zeros((1,Nvar))
        for tv in np.arange(Nvar):
            score2[tv] = []
            m_ind = [tv]
            Xm = np.zeros_like(X3)        
            Xm[:,m_ind] = 1
            X4 = X3*Xm
            X2 = np.column_stack([np.ones_like(y),X4])
            ss= ShuffleSplit(n_splits=k, test_size=0.25, random_state=0)
            y2 = ndimage.gaussian_filter(y,0)
            cv_results = cross_validate(reg, X4, y2, cv = ss , 
                                        return_estimator = True, 
                                        scoring = 'explained_variance')
            score2[tv] = np.concatenate((score2[tv], cv_results['test_score']))
            mi_score[0,tv] = np.median(cv_results['test_score'])
        maxS = np.argmax(mi_score)
        maxscore = score2[maxS]
        
        
        # ==== Adding task  ====
        it = 0
        score2 = {}
        while it < Nvar:
            for tv in np.arange(Nvar):
                score2[tv] = []
                m_ind = np.unique(np.append(maxS,tv))
                Xm = np.zeros_like(X3)        
                Xm[:,m_ind] = 1
                X4 = X3*Xm
                X2 = np.column_stack([np.ones_like(y),X4])
                ss= ShuffleSplit(n_splits=k, test_size=0.25, random_state=0)
                y2 = ndimage.gaussian_filter(y,0)
                cv_results = cross_validate(reg, X4, y2, cv = ss , 
                                                return_estimator = True, 
                                                scoring = 'explained_variance')
                score2[tv] = np.concatenate((score2[tv], cv_results['test_score']))
                mi_score[0,tv] = np.median(cv_results['test_score'])
            maxS2 = np.argmax(mi_score)
            maxscore2 = score2[maxS2]
            s,p = stats.ks_2samp(maxscore, maxscore2, alternative = 'less')
            if p > 0.05:
                it = Nvar
            else:
                maxscore = maxscore2
                maxS = np.unique(np.append(maxS,maxS2))
                it += 1
                
        # === Finalizing m_ind==
        
        
        
        Xm = np.zeros_like(X3)        
        Xm[:,maxS] = 1
        X4 = X3*Xm
        X2 = np.column_stack([np.ones_like(y),X4])
        ss= ShuffleSplit(n_splits=k, test_size=0.25, random_state=0)
        y2 = ndimage.gaussian_filter(y,0)
        cv_results = cross_validate(reg, X4, y2, cv = ss , 
                                    return_estimator = True, 
                                    scoring = 'explained_variance')        
        theta = np.zeros((np.size(X2,1)-1,k))
        inter = np.zeros((1,k))
        pp = 0
        for model in cv_results['estimator']:
            theta[:,pp] = model.coef_ 
            inter[:,pp] = model.intercept_
            pp = pp+1
        theta3 = np.concatenate((np.mean(inter,1),np.mean(theta,1)))
        yhat = X2 @theta3
        # yhat1 = X2[0:200,:] @ theta3
        # yhat2 = X2[200:,:] @ theta3
        
        
        score = np.concatenate((score, cv_results['test_score']))
        TT2 = np.concatenate((TT2,np.mean(theta,1)))
        Intercept = np.concatenate((Intercept,np.mean(inter,1)))
        CI2 = np.concatenate((CI2,stats.sem(theta,1)))

        Yhat = np.concatenate((Yhat,yhat))
        # Yhat1 = np.concatenate((Yhat1,yhat1))
        # Yhat2 = np.concatenate((Yhat2,yhat2))
        
        
    Yhat = np.reshape(Yhat,(int(t_period/window),N_trial2)).T
    # Yhat1 = np.reshape(Yhat1,(int(t_period/window),N_trial2)).T
    # Yhat2 = np.reshape(Yhat2,(int(t_period/window),N_trial2)).T
    
    
    TT2 = np.reshape(TT2,(int(t_period/window),np.size(X3,1))).T
    CI2 = np.reshape(CI2,(int(t_period/window),np.size(X3,1))).T
    score = np.reshape(score,(int(t_period/window),k))
    
    
    
    
    # Figures
    if fig_on ==1:
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))
        
            
        if c_ind == 0:
            cmap = ['tab:purple', 'tab:orange', 'tab:green','tab:blue']
            clabels = ["contin","action","correct","stim"]
        elif c_ind == -1:
            cmap = ['tab:purple', 'tab:orange', 'tab:green','tab:blue','tab:olive']
            clabels = ["contin","action","correct","stim","history"]
        elif c_ind == 2 or c_ind ==3 or c_ind == 1:
            cmap = ['tab:purple','tab:blue','tab:red','tab:olive']
            clabels = ["Contingency","stim","reward","history",]
            lstyles = ['solid','solid','solid','solid']

            # cmap = ['tab:orange','tab:purple','tab:blue','tab:red','tab:olive','tab:olive']
            # clabels = ["lick","Contingency","stim","reward","history","history2"]
            # lstyles = ['solid','solid','solid','solid','solid','dashed']

            
            
            
        x_axis = np.arange(1,t_period,window)
        for c in range(np.size(X3,1)):        
            ax2.plot(x_axis,ndimage.gaussian_filter(TT2[c,:],2),linewidth = 2.0,
                     color = cmap[c], label = clabels[c], linestyle = lstyles[c])
            ax2.fill_between(x_axis,(ndimage.gaussian_filter(TT2[c,:],2) - CI2[c,:]),
                            (ndimage.gaussian_filter(TT2[c,:],2 )+ CI2[c,:]), color=cmap[c], alpha = 0.2)
        
        # ax2.legend(loc = 'upper right')
    
        # e_lines = np.array([0,500,500+int(D_ppc[n,3]),2500+int(D_ppc[n,3])])
        e_lines = np.array([0,500,500+1000,2500+1000])
        # e_lines = np.array([0,500,500+1000,2500+1000, 8000, 8000+500, 8000+1500, 8000+ 3500])

        e_lines = e_lines+prestim
    
        
        ax2.vlines(x =e_lines, 
                  ymin = np.amin(ndimage.gaussian_filter(TT2,sigma = [0,3])), 
                  ymax = np.amax(ndimage.gaussian_filter(TT2,sigma = [0,3])),
                  linestyles = 'dashed',
                  colors = 'black', 
                  linewidth = 2.0)
        
        ax4.plot(x_axis,ndimage.gaussian_filter(np.mean(score,1)*1e2,1))
        
        var_top = min(max(ndimage.gaussian_filter(np.mean(score,1)*1e2,1)),100)
            
        # Plotting firing rates for one condition VS the other
        # 0 : contingency 
        # 1 : lick vs no lick
        # 2 : correct vs wrong
        # 3 : stim 1 vs stim 2
        # if c_ind ==0:
        #    # stim_ind = X3[:,3] == 1 
        # else:
        stim_ind1 = X[:,2] == 1     
        stim_ind2 = X[:,2] == 0  
        
    
        ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind1,:],0),0),
                  linewidth = 2.0, color = "blue",label = '10kHz',linestyle = lstyles[3])
        # ax1.fill_between(x_axis,(ndimage.gaussian_filter(np.mean(Y[stim_ind1,:],0),2) - ndimage.gaussian_filter(np.std(Y[stim_ind1,:],0),2)),
        #                 (ndimage.gaussian_filter(TT2[c,:],2 )+ CI2[c,:]), color=cmap[c], alpha = 0.2)
        ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind2,:],0),0),
                  linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax1.set_title('Firing rate y')
        ax1.legend(loc = 'upper right')
    
        
        # ax3.plot(x_axis,ndimage.gaussian_filter(np.mean(Yhat[stim_ind1,:],0),2),
        #           linewidth = 2.0, color = cmap[3],linestyle = lstyles[3])
        # ax3.plot(x_axis,ndimage.gaussian_filter(np.mean(Yhat[stim_ind2,:],0),2),
        #           linewidth = 2.0, color = cmap[3],linestyle = lstyles[4]) 
        # ax3.set_title('Prediction y_hat')
    
        ax2.set_title('unit_'+str(n+1))
        ax4.set_title('explained variance')
        ax4.set_ylim(bottom = -2, top = var_top)
        plt.savefig("eg_units"+ str(n+1) + ".svg")

        plt.show()

    Model_Theta = TT2
    
    return X, Y, Yhat, Model_Theta, score, Intercept

# %% Main
        


# %% Model analysis, categorizing each neuron
""" 
Data{score} = 100 by k
Data{coef}  = n_x by 100 where n_x is the number of variables
window2      :   score and weight coefs are binned by a moving window 
                 with step size bin_size and window size window2 

OUTPUT
max_ind     :   best index for peak of score(explained variance) (not in ms)
best_score  :   average score at max_ind
coef        :   Weight coefficients at max_ind
model_mean  :   Weight coefficients across t_period
   
"""
def Model_analysis(n,window, window2,Data,c_ind,ana_period):
    
    # time currently defined by window size* data size. ana_period should also be defined thus 
    bin_size = int(window2/window)
    ana_bin = ana_period/(window2/2)

    Dat_length  = np.size(Data[n,c_ind-1]["score"],0)
    Model_Theta = Data[n,c_ind-1]["coef"]/(np.max(np.abs(Data[n,c_ind-1]["coef"]))+1) # Soft normalization

    score_mean  = np.zeros((1,2*int(Dat_length/bin_size)))
    score_pool  = np.zeros((np.size(Data[n,c_ind-1]["score"],1),2*int(Dat_length/bin_size)))
    score_var   = np.zeros((1,2*int(Dat_length/bin_size)))
    model_mean  = np.zeros((np.size(Model_Theta,0),2*int(Dat_length/bin_size)))
    
    k = 0;
    for ind in np.arange(0,Dat_length-bin_size/2,int(bin_size/2)):
        ind = int(ind)
        score_pool[:,k] = np.mean(Data[n,c_ind-1]["score"][ind:ind+bin_size,:],0)
        score_mean[0,k] = np.mean(Data[n,c_ind-1]["score"][ind:ind+bin_size,:])
        score_var[0,k]  = np.var(Data[n,c_ind-1]["score"][ind:ind+bin_size,:])
        model_mean[:,k] = np.mean(np.abs(Model_Theta[:,ind:ind+bin_size]),1)
        k = k+1
    
    max_ind = np.argmax(score_mean[0,int(ana_bin[0]):int(ana_bin[1])]) + int(ana_bin[0])
    best_score = score_mean[0,max_ind]
    coef = model_mean[:,max_ind]
    
    
    return max_ind, best_score, coef, model_mean, score_mean, score_var, score_pool

# %% 

Nvar = 4

def build_model(n, t_period, prestim, window,k,c_ind,ca):
    for m_ind in np.arange(Nvar):
        X, Y, Yhat, Model_Theta, score, intercept = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,m_ind,0)
        Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score, 'Y' : Y,'Yhat' : Yhat}
        mi, bs, coef,beta_weights,mean_score, var_score,score_pool = Model_analysis(n, window, window2, Data,c_ind,ana_period)
        S[0,m_ind] = mean_score[0,mi]
        mean_score[mean_score<weight_thresh] = 0
        DataS[n,c_ind-1,m_ind] = {"mean_score" : mean_score, "var_score" : var_score,"score_pool" : score_pool}
    maxS = np.argmax(S)
    max_score_pool = DataS[n,c_ind-1,maxS]["score_pool"]
    
    it = 0
    while it < Nvar:
        p = np.zeros((np.size(X,1),np.size(max_score_pool,1)))
        mean_score_pool = np.zeros((np.size(X,1),np.size(max_score_pool,1)))
        if np.any(DataS[n,c_ind-1,np.argmax(S)]["mean_score"] >  DataS[n,c_ind-1,np.argmax(S)]["var_score"]):
            for m_ind in np.arange(Nvar):
                m_ind2 = np.unique(np.append(maxS,m_ind))
                X, Y, Yhat, Model_Theta, score, intercept = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,m_ind2,0)
                Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score, 'Y' : Y,'Yhat' : Yhat}
                mi, bs, coef,beta_weights,mean_score, var_score ,score_pool = Model_analysis(n, window, window2, Data,c_ind,ana_period)
                S[0,m_ind] = mean_score[0,mi]
                mean_score[mean_score<weight_thresh] = 0
                DataS[n,c_ind-1,m_ind] = {"mean_score" : mean_score, "var_score" : var_score,"score_pool" : score_pool}
                mean_score_pool[m_ind,:] = mean_score
                for t in np.arange(np.size(max_score_pool,1)):
                    s,p[m_ind,t] = stats.ks_2samp(score_pool[:,t], max_score_pool[:,t], alternative = 'less')
        
        p = p<0.05
        if np.any(p) == True:
            T = mean_score_pool-np.mean(max_score_pool,0) 
            T = T*p

            maxS = np.unique(np.append(maxS,np.argmax(np.max(T,1))))
            max_score_pool = DataS[n,c_ind-1,np.argmax(np.max(T,1))]["score_pool"]
            it += 1
        else:
            it = Nvar   
    
    return maxS


# %% Run main GLM code
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
    
# %% Run GLM 
Data = {}
Data = np.load('Data_Task_1012.npy', Data,allow_pickle= True).item()
# 
# additional code for explained variance comparison
DataS = {}
S = np.zeros((1,Nvar))
ana_period = np.array([0, t_period+prestim])
weight_thresh = 2*1e-2

# change c_ind and n here. 

for c_ind in c_list:
    # t = 0 
    good_list2 = [];
    for n in np.arange(np.size(D_ppc,0)):
        
        n = int(n)
        # X, Y, Yhat, Model_Theta, score = glm_per_neuron(n, t_period, prestim, window,k,c_ind)
        # Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score} 
        try:
            # maxS = build_model(n, t_period, prestim, window, k, c_ind, ca)
            # maxS = Data[n,c_ind-1]["maxS"]
            # maxS = [0,1,2,3]   
            X, Y, Yhat, Model_Theta, score, intercept = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,1)
            Data[n,c_ind-1] = {"X":X,"coef" : Model_Theta, "score" : score, 'Y' : Y,'Yhat' : Yhat, "intercept" : intercept}
            # t += 1
            # print(t,"/",len(good_list))
            good_list2 = np.concatenate((good_list2,[n]))
            print(n)
            
        except KeyboardInterrupt:
            
            break
        except:
            
            print("Error, probably not enough trials") 
# np.save('transition_1101.npy', Data,allow_pickle= True)      
# 

# %% testing model weight stuff

fig, axes = plt.subplots(2,2,figsize = (10,8))

ymean = np.zeros((1,160))
ymean[0,:] = intercept
x_axis = np.arange(1, prestim+t_period, window)
axes[0,0].plot(x_axis,np.mean(Y[X[:,1]==1,:],0), linestyle = 'solid')
axes[0,0].plot(x_axis,np.mean(Y[X[:,1]==0,:],0), linestyle = 'dotted')
axes[1,0].plot(x_axis,ymean[0,:],c = "black")
# axes[1,0].plot(x_axis,np.mean(Yhat[X[:,1]==1,:],0), linestyle = 'solid')
# axes[1,0].plot(x_axis,np.mean(Yhat[X[:,1]==0,:],0), linestyle = 'dotted')



theta3 = np.concatenate((ymean,Model_Theta),0)
X2 = np.concatenate((np.ones((np.size(X,0),1)),X),1)

yhat2 = X2[:,[0,2]] @ theta3[[0,2],:]

yhat3 = X2[:,[0,3]] @ theta3[[0,3],:]

axes[0,1].plot(x_axis,np.mean(yhat2[X[:,1]==1,:],0), c = "tab:blue",linestyle = 'solid')
axes[0,1].plot(x_axis,np.mean(yhat2[X[:,1]==0,:],0),c = "tab:blue", linestyle = 'dotted')
# axes[0,1].plot(x_axis,intercept)

axes[1,1].plot(x_axis,np.mean(yhat3[X[:,2]==1,:],0),c = "tab:red", linestyle = 'solid')
axes[1,1].plot(x_axis,np.mean(yhat3[X[:,2]==0,:],0),c = "tab:red", linestyle = 'dotted')

# %% for each weight, get corresponding yhat
# Calculating R2 per neuron
# good_list = good_list_int
good_list = np.arange(np.size(D_ppc,0))
d_list = good_list > 179
# d_list = good_list > 118
d_list3 = good_list <= 179
# d_list3 = good_list <= 118

c_ind = 1
ax_sz = 4

good_list_sep = good_list[d_list3]

# Rscore = {}
Rscore = np.zeros((ax_sz+1,np.size(good_list)))
    
y_lens = np.arange(160)
   
for n in np.arange(np.size(good_list,0)):
        # print(n)
    nn = good_list[n]
    nn = int(nn)
    # maxS = Data[nn,c_ind-1]["maxS"]
    try:
        X = Data[nn,c_ind-1]["X"]
        intercept = Data[nn,c_ind-1]["intercept"]
        # X, Y, Yhat, Model_Theta, score, intercept = glm_per_neuron(nn, t_period, prestim, window,k,c_ind,ca,maxS,0)


    except:                
        X, Y, Yhat, Model_Theta, score, intercept = glm_per_neuron(nn, t_period, prestim, window,k,c_ind,ca,1)
        # Data[nn,c_ind-1] = {"X" : X,"coef" : Model_Theta, "intercept" : intercept, "score" : score, 'Y' : Y,'Yhat' : Yhat, 'maxS': maxS}
        
    Y = Data[nn,c_ind-1]["Y"][:,:]
    Yhat = Data[nn,c_ind-1]["Yhat"][:,:]
    Model_Theta = Data[nn,c_ind-1]["coef"]
    ymean = np.ones((len(y_lens),np.size(X,0))).T*intercept
        
    theta3 = np.concatenate(([ymean[0,:]],Model_Theta[:,y_lens]),0)
    X2 = np.concatenate((np.ones((np.size(X,0),1)),X),1)
        
    for f in np.arange(ax_sz):
        yhat2 = X2[:,[0,f+1]] @ theta3[[0,f+1],:]
        Rscore[f,n] = 1- np.sum(np.square(Y-yhat2))/np.sum(np.square(Y-ymean))
        if Rscore[f,n] ==0:
            Rscore[f,n] = -1
                
    Rscore[ax_sz,n] = 1- np.sum(np.square(Y-Yhat))/np.sum(np.square(Y-ymean))
        # Rscore[c_ind][:,n]    

# scatter_ind = [np.arange(ax_sz+1)]*np.ones((ax_sz+1,len(good_list))).T
# scatter_ind = scatter_ind.T

# %% Plot example neuron figures
cmap3 = ['tab:purple','tab:blue','tab:red','tab:olive']

n = 101
c_ind = -2
Y = Data[n, c_ind -1]["Y"]
Yhat = Data[n,c_ind-1]["Yhat"]
Model_Theta = Data[n,c_ind-1]["coef"]
intercept = Data[n,c_ind-1]["intercept"]

X = Data[n,c_ind-1]["X"]
X2 = D_ppc[n,2][:,2:6] # task variables
X2 = np.concatenate((X2[0:200,:],X2[D_ppc[n,4][0][0]:,:]),0)

fig, axes = plt.subplots(4,1, figsize = (8,20))
x_axis = np.arange(1, prestim+t_period, window)
x_axis = x_axis*1e-3-1
for f in np.arange(4):
    axes[f].plot(x_axis,ndimage.gaussian_filter(Model_Theta[f,:],2),c = cmap3[f])

plt.savefig("eg_units_TV_weights.svg")

trial_ind = {}
# trial_ind[0] = (X2[:,0] >0)  * (X2[:,1] >0) # hit
# trial_ind[1] = (X2[:,0] == 0) * (X2[:,1] >0) # FA
# trial_ind[2] = (X2[:,0] == 0) * (X2[:,1] == 0) # CR
# trial_ind[3] = (X2[:,0] >0) * (X2[:,1] == 0) # Miss


trial_ind[0] = (X2[:,0] >0)  * (X2[:,3] >0) # R1 go
trial_ind[1] = (X2[:,0] == 0) * (X2[:,3] ==0) # R1 no-go
trial_ind[2] = (X2[:,0] > 0) * (X2[:,3] == 0) # R2 go
trial_ind[3] = (X2[:,0] == 0) * (X2[:,3] > 0) # R2 no-go

fig, axes = plt.subplots(1,1, figsize = (10,8))
for f in np.arange(4):
    axes.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[trial_ind[f],:],0),2),c = cmap3[f])
plt.savefig("eg_units_Y.svg")

fig, axes = plt.subplots(1,1, figsize = (10,8))
for f in np.arange(4):
    axes.plot(x_axis,ndimage.gaussian_filter(np.mean(Yhat[trial_ind[f],:],0),2),c = cmap3[f])
plt.savefig("eg_units_Yhat.svg")

fig, axes = plt.subplots(1,1, figsize = (10,8))
axes.plot(x_axis,ndimage.gaussian_filter(intercept,2),c = cmap3[f])
plt.savefig("eg_units_bias.svg")


for nn in [101]:
    maxS = Data[nn,c_ind-1]["maxS"]
    X, Y, Yhat, Model_Theta, score, intercept = glm_per_neuron(nn, t_period, prestim, window,k,c_ind,ca,maxS,1)

        
# %% bar plot
# PPC_AC = [81 57 57 85]

# b1 = [129, 59,  107, 72]
# b1 = np.array(b1)/420
# b2 = [67,  34,  51, 27]
# b2 = np.array(b2)/180
# b3 = [9, 37, 37, 77]

# IC
# b11 = [13, 26, 54, 56]
# b11 = np.array(b11)/(116)

# b12 = [23, 7, 27, 16]
# b12 = np.array(b12)/(116)

# b21 = [16, 3, 33, 48]
# b21 = np.array(b21)/(103)

# b22 = [31, 8, 12, 24]
# b22 = np.array(b22)/(103)

# b31 = [11, 1, 51, 58]
# b31 = np.array(b31)/(103)

# b32 = [17, 15, 15, 13]
# b32 = np.array(b32)/(103)

b11 = [55, 25, 104, 133]
b11 = np.array(b11)/(338)

b12 = [59, 14, 70, 88]
b12 = np.array(b12)/(338)

b21 = [57, 31, 52, 106]
b21 = np.array(b21)/(246)

b22 = [38, 12, 50, 61]
b22 = np.array(b22)/(246)

b31 = [46, 9, 99, 110]
b31 = np.array(b31)/(316)

b32 = [49, 17, 58, 90]
b32 = np.array(b32)/(316)


# # TTR
# b2 = [114, 88, 126, 103]
# b2 = np.array(b2)/423

cmap3 = ['tab:pink','tab:blue','tab:red','tab:orange']

fig, axes = plt.subplots(2,1,figsize = (10,10), sharex = True)
fig.tight_layout()
fig.subplots_adjust(hspace=0)

axes[0].bar(np.arange(4)*3,b11, color = cmap3, alpha = 0.5, width = 0.5)
axes[0].bar(np.arange(4)*3+0.7,b21, color = cmap3, alpha = 1, width = 0.5)
axes[0].bar(np.arange(4)*3+1.4,b31, color = cmap3, alpha = 0.5, width = 0.5)
axes[0].set_ylim([0,0.6])

axes[1].bar(np.arange(4)*3,-b12, color = cmap3, alpha = 0.5, width = 0.5)
axes[1].bar(np.arange(4)*3+0.7,-b22, color = cmap3, alpha = 1, width = 0.5)
axes[1].bar(np.arange(4)*3+1.4,-b32, color = cmap3, alpha = 0.5, width = 0.5)
axes[1].set_ylim([-0.6,0])
# plt.savefig("nbTVunitsPAC2.svg")

# axes.set_ylim([0, 0.8])
# nb_TV = [];
# for n in good_listRu[160:]:
    
#     maxS = Data[n,2]["maxS"]
#     try:
#         nb_TV.append(np.size(maxS,0))
#     except:
#         nb_TV.append(np.size(maxS))


# plt.hist(nb_TV)



# %% plot R score


cmap = ['tab:purple','tab:blue','tab:red','tab:orange']
# c_ind = -1

# d_list = good_list > 118
d_list3 = good_list <= 179
# d_list3 = good_list <= 118

d_list = good_list > 179


def make_RS(d_list):
    fig, axes = plt.subplots(1,1, figsize = (10,8))
    Rsstat = {}
    for f in np.arange(0,ax_sz):
        Rs = Rscore[f,d_list]
        Rmax = Rscore[4,d_list]
        Rmax = Rmax[Rs>0.02]
        Rs = Rs[Rs>0.02]
    
        # Rs = Rs/(Rmax+0.03)
        Rsstat[c_ind,f] = Rs
        axes.scatter(np.ones_like(Rs)*(f+(c_ind+1)*-0.3),Rs,c = cmap[f])
        axes.scatter([(f+(c_ind+1)*-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')
            # axes.boxplot(Rs,positions= [f+(c_ind+1)*-0.3])
    axes.scatter(np.ones_like(Rscore[4,d_list])*(4+(c_ind+1)*-0.3),Rscore[4,d_list])
    axes.scatter([(4+(c_ind+1)*-0.3)],np.mean(Rscore[4,d_list]),c = 'k',s = 500, marker='_')
        
    Rsstat[c_ind,4] = Rscore[4,d_list]
    
        # axes.boxplot(Rscore[c_ind][4,d_list3],positions= [4+(c_ind+1)*-0.3])
    axes.set_ylim([-0.05,0.3])
    plt.savefig("PPC_AC.svg")

    return Rsstat


RsStat_PIC = make_RS(d_list3)
RsStat_PAC = make_RS(d_list)



bins = np.arange(0,0.25,0.0025)
fig, axes = plt.subplots(1,1,figsize = (5,5))
axes.hist(RsStat_PAC[c_ind,4], bins , alpha=0.7, rwidth=1)
axes.hist(RsStat_PIC[c_ind,4], bins , alpha=0.7, rwidth=1)

np.mean((RsStat_PAC[c_ind,4][RsStat_PAC[c_ind,4]>0.02]))
# 

# stats.ks_2samp(RsStat_PAC[-1,4],RsStat_PAC2[-1,4])
plt.savefig("Rscore_hist.svg")
# 


good_listR = Rscore[4,:] > 0.02
good_listR[22] = False
good_listR[50] = False
# good_listR[12] = False
good_listRu = good_list[good_listR]


# %% Normalized population average of task variable weights

# d_list = good_listRu > 179

# d_list3 = good_listRu <= 179

# # c_ind = 3
# # good_list2 = good_list[d_list & good_listR]

# # cat_list = best_kernel[c_ind][0,:] != 0 # Only neurons that were categorized

# # good_list_sep = good_list[cat_list]
# # good_list_sep = good_list[d_list & good_listR]
# good_list_sep = good_listRu[:]
# # good_list_sep = good_list


# weight_thresh = 4*1e-2


# if c_ind == 0 or c_ind == -2:
#     cmap3 = ['tab:purple','tab:blue','tab:red','tab:olive']
#     ax_sz = len(cmap3)
#     clabels = ["lick","Contingency","stim","reward","history"]
#     lstyles = ['solid','solid','solid','solid','solid']
    

# score = np.zeros((160,1))
# Convdata = {}
# norm_score_all = {};
# norm_score_all = np.zeros((np.size(good_list_sep),np.size(score,0)))
# for b_ind in np.arange(ax_sz):
#     Convdata[b_ind] = np.zeros((np.size(good_list_sep),np.size(score,0)))
        
# for n in np.arange(np.size(good_list_sep,0)):
#     # n = int(n)
#     nn = int(good_list_sep[n])
#     Model_coef = Data[nn, c_ind-1]["coef"]
#     Model_score = Data[nn, c_ind-1]["score"]
#     X = Data[nn,c_ind-1]["X"]
#     bias = Data[nn,c_ind-1]["intercept"]
#     # Model_coef = np.abs(Model_coef)/(np.max(np.abs(Model_coef)) + 0.1) # soft normalization value for model_coef
#     Model_coef = Model_coef/(np.max(np.abs(Model_coef)) + 0.2) # soft normalization value for model_coef
# # 
#     norm_score = np.mean(Model_score, 1)
#     norm_score[norm_score < weight_thresh] = 0
#     norm_score = ndimage.gaussian_filter(norm_score,1)
#     norm_score[norm_score > 0] = 1 
#     # if np.max(norm_score)>0:
#     #     norm_score = norm_score/(np.max(norm_score)+weight_thresh)
#     # else:
#     #     norm_score = 0    
    
#     # if good_listR[n] == True:
#     # norm_score = ndimage.gaussian_filter(norm_score,4)
#     conv = Model_coef*norm_score
#     # else: 
#         # conv = Model_coef*0
#     # if np.mean(norm_score*norm_score*1e4) > weight_thresh*1e2:
#     #     conv = Model_coef
#     # else:
#     #     conv = Model_coef*0
    
#     # norm_score_all[n,:] = norm_score.T
#     for b_ind in np.arange(np.size(Model_coef, 0)):
#         Convdata[b_ind][n, :] = conv[b_ind, :]


# x_axis = np.arange(1, prestim+t_period, window)
# fig, axes = plt.subplots(1,1,figsize = (7,4))

# for f in range(ax_sz):
#         error = np.std(Convdata[f],0)/np.sqrt(np.size(good_list_sep))
#         y = ndimage.gaussian_filter(np.mean(Convdata[f],0),2)
#         axes.plot(x_axis*1e-3-prestim*1e-3,y,c = cmap3[f],linestyle = lstyles[f])
#         axes.fill_between(x_axis*1e-3-prestim*1e-3,y-error,y+error,facecolor = cmap3[f],alpha = 0.3)
#         axes.set_ylim([-0.15,0.25])

# # axes[1].plot(x_axis*1e-3-prestim*1e-3,ndimage.gaussian_filter(np.mean(norm_score_all,0),2))
# # plt.savefig("PSTHPPC_AC.svg")

# e_lines = np.array([0, 500, 500+1000, 2500+1000])
# e_lines = e_lines+500









# %% Normalized population average of task variable weights
c_ind = 3
d_list = good_list > 179
# d_list3 = good_list <= 179
d_list3 = good_list <= 179
# good_list_sep = np.arange(600)
# good_list_sep = np.arange(336)
# c_ind = 2
# good_list2 = good_list[d_list & good_listR]

# cat_list = best_kernel[c_ind][0,:] != 0 # Only neurons that were categorized 

# good_list_sep = good_list[cat_list]
# good_list_sep = good_list[d_list]
Lic = np.where(good_listRu <180)
Lic = Lic[0][-1]
good_list_sep = good_listRu[:]



weight_thresh = 5*1e-2


# if c_ind == 3 or c_ind == -2:
cmap3 = ['tab:purple','tab:blue','tab:red','tab:olive']
ax_sz = len(cmap3)
    # cmap3 = [,'tab:blue','tab:red','tab:olive']
    # ax_sz = len(cmap3)
clabels = ["lick","Contingency","stim","reward","history"]
lstyles = ['solid','solid','solid','solid','solid']
    

score = np.zeros((160,1)) 
Convdata = {}
norm_score_all = {};
norm_score_all = np.zeros((np.size(good_list_sep),np.size(score,0)))
for b_ind in np.arange(ax_sz):
    Convdata[b_ind] = np.zeros((np.size(good_list_sep),np.size(score,0)))
        
for n in np.arange(np.size(good_list_sep,0)):
    # n = int(n)
    nn = int(good_list_sep[n])
    Model_coef = Data[nn, c_ind-1]["coef"]
    Model_score = Data[nn, c_ind-1]["score"]

    # Model_coef = np.abs(Model_coef)/(np.max(np.abs(Model_coef)) + 0.1) # soft normalization value for model_coef
    Model_coef = Model_coef/(np.max(np.abs(Model_coef)) + 0.2) # soft normalization value for model_coef
# 
    norm_score = np.mean(Model_score, 1)
    norm_score[norm_score < weight_thresh] = 0
    norm_score = ndimage.gaussian_filter(norm_score,2)
    norm_score[norm_score > 0] = 1 
    # if np.max(norm_score)>0:
    #     norm_score = norm_score/(np.max(norm_score)+weight_thresh)
    # else:
    #     norm_score = 0    
    
    # if good_listR[n] == True:
    # norm_score = ndimage.gaussian_filter(norm_score,4)
    conv = Model_coef*norm_score
    # conv = Model_coef
    # else: 
        # conv = Model_coef*0
    # if np.mean(norm_score*norm_score*1e4) > weight_thresh*1e2:
    #     conv = Model_coef
    # else:
    #     conv = Model_coef*0
    
    # norm_score_all[n,:] = norm_score.T
    for b_ind in np.arange(np.size(Model_coef, 0)):
        Convdata[b_ind][n, :] = conv[b_ind, :]


x_axis = np.arange(1, t_period+prestim, window)
fig, axes = plt.subplots(1,1,figsize = (10,8))

for f in range(ax_sz):
        error = np.std(Convdata[f],0)/np.sqrt(np.size(good_list_sep))
        y = ndimage.gaussian_filter(np.mean(Convdata[f],0),2)
        y = np.abs(y)
        axes.plot(x_axis*1e-3-prestim*1e-3,y,c = cmap3[f],linestyle = lstyles[f])
        axes.fill_between(x_axis*1e-3-prestim*1e-3,y-error,y+error,facecolor = cmap3[f],alpha = 0.3)
        axes.set_ylim([-0.02,0.20])

# axes[1].plot(x_axis*1e-3-prestim*1e-3,ndimage.gaussian_filter(np.mean(norm_score_all,0),2))

e_lines = np.array([0, 500, 500+1000, 2500+1000])
e_lines = e_lines+500


# 
# np.save('Conv_R2_350.npy', Convdata,allow_pickle= True)      


# %% Calculating weights

weight = {}
p = {}
p[-1] = np.arange(30,60)
# p[-1] = np.arange(50,80)

p[-2] = p[-1]
# p[-2] = np.arange(0,15)

Lg = len(good_listRu)
# Lg = 256
# Lac = np.sum(d_list)
# Lic = np.sum(d_list3)    
Lic = np.where(good_listRu <180)
Lic = Lic[0][-1]
for f in np.arange(ax_sz):  
    weight[-1,f]= np.zeros((1,Lg))
    weight[-2,f] = np.zeros((1,Lg))    
    for c_ind in [-1,-2]:
        if c_ind == -1:
            for n in np.arange(Lic,Lg):
                weight[c_ind,f][0,n] = np.mean(Convdata[f][n,p[c_ind]])
        if c_ind == -2:
            for n in np.arange(Lic):
                weight[c_ind,f][0,n] = np.mean(Convdata[f][n,p[c_ind]])
                
                
# fig, axes = plt.subplots(1,1,figsize = (10,8))
# axes.scatter(weight[-1,2],-weight[-2,2])
# axes.set_xlim([-0.5,0.5])
# axes.set_ylim([-0.5,0.5])
# weightr2hist = weight

# %% Comparing weights between two time periods
weight = {}
p = {}
Lg = len(good_listRu) # 532
# Lg = len(good_listRu)
Lic = np.where(good_listRu <180)
Lic = Lic[0][-1]
# d_list = good_list > 118
# d_list3 = good_list <= 179
# d_list3 = good_list <= 118
# Lg = 600
p[0] = np.arange(50,80)
p[1] = np.arange(80,100)

for f in np.arange(ax_sz):
    for z in [0,1]:
        weight[z,f] = np.zeros((1,Lg))
        for n in np.arange(Lg):
            weight[z,f][0,n] = np.mean(Convdata[f][n,p[z]])
            


f1 = 0
f2 = 0
fig, axes = plt.subplots(1,1,figsize = (7,7))
axes.scatter(weight[0,f1][0,Lic:Lg],weight[1,f2][0,Lic:Lg], c = "blue")
# axes.scatter(weight[0,f1][0,0:Lic],weight[1,f2][0,0:Lic], c = "red")
axes.vlines([-0.1, 0.1],-1,1,color = "black", linestyles = "dotted")
axes.hlines([-0.1, 0.1],-1,1,color = "black", linestyles = "dotted")

axes.set_xlim([-0.8,0.8])
axes.set_ylim([-0.8,0.8])
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
	label.set_fontsize(16)
    
    


rg = np.arange(Lic,Lg)
# rg = np.arange(Lic)
th = 0.1
  
list2 = (np.abs(weight[0,f1][0,rg]) > th) * (np.abs(weight[1,f2][0,rg]) > th)
# list3=  np.logical_or(np.abs(weight[0,f1][0,rg]) > th,np.abs(weight[1,f2][0,rg]) > th)
list3=  (np.abs(weight[0,f1][0,rg]) > th)* (np.abs(weight[1,f2][0,rg]) < th)
list4=  (np.abs(weight[1,f2][0,rg]) > th)* (np.abs(weight[0,f1][0,rg]) < th)


# list4 = (np.abs(weight[0,f1][0,rg]) < th) * (np.abs(weight[1,f2][0,rg]) < th)

# common vs indy
print(np.sum(list2))    
print(np.sum(list3))
print(np.sum(list4))

# print(np.sum(list4))
# good_listRu[list2]


# print(np.sum(list3*list2))

# %% calculate correlation for weight distribution

f = 1
W1 = weight[0,f][0,0:Lic]
W2 = weight[1,f][0,0:Lic]
W3 = weight[0,f][0,Lic:Lg]
W4 = weight[1,f][0,Lic:Lg]



# W1a = W1[(W1!=0)*(W2!=0)]
# W2a = W2[(W1!=0)*(W2!=0)]

# W3a = W3[(W3!=0)*(W4!=0)]
# W4a = W3[(W3!=0)*(W4!=0)]


# W1a = W1[(W1>0.1)]
# W2a = W1[(W1<-0.1)]

# W3a = W3[(W3>0.1)]
# W4a = W3[(W3<-0.1)]


W1a = np.abs(W1a)
W2a = np.abs(W2a)
W3a = np.abs(W3a)
W4a = np.abs(W4a)

import seaborn as sns

sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
fig, ax = plt.subplots(1,1,figsize = (7,7))

# ax = sns.boxplot(data=(W1a,W3a,W2a,W4a), showfliers = False)
ax = sns.stripplot(data=(W1a,W3a,W2a,W4a), palette=["red","blue","red","blue"])
ax = sns.pointplot(data=(W1a,W3a,W2a,W4a), color="black", join = False)

ax.set_ylim([-0.9,0.9])

plt.show()

stats.ks_2samp(W2a,W4a )

# %% test weight distribution 
# weight1 = weight
bins = np.arange(0,1,0.05)
fig, axes = plt.subplots(1,1,figsize = (10,8))
axes.hist(np.abs(weight[1,2][0,:]), bins , alpha=0.7, rwidth=0.85)
# axes.hist(np.abs(weight1[1,2][0,:]), bins , alpha=0.7, rwidth=0.85)
axes.set_xlim([-0.05,0.85])
axes.set_ylim([0,50])
# np.percentile(np.abs(weight[1,2][0,:]),99, interpolation='lower')

# %%




f = 1
list2 = (np.abs(weight[-1,f]) > 0.1) #* (weight[-2,f] == 0)
list3 = (np.abs(weight[-2,f]) > 0.1) #* (weight[-2,f] == 0)

C1 = Convdata[f][list2[0,:],:]
C1 = C1[np.max(np.abs(C1),1)>0,:]
C2 = Convdata[f][list3[0,:],:]
C2 = C2[np.max(np.abs(C2),1)>0,:]

C1 = np.abs(np.mean(C1[:,0:20],1))
C2 = np.abs(np.mean(C2[:,0:20],1))

stats.ks_2samp(C1,C2)


# %%

f =1
# list2 = (weight[-1,f] > 0.1)# or (weight[-2,f] < -0.1)
# list3 = (weight[-2,f] > 0.1)# or (weight[-2,f] < -0.1)

list2 = (np.abs(weight[-1,f]) > 0.1)
list3 = (np.abs(weight[-2,f]) > 0.1)
nAC = 162 
nIC = 60
# print(np.sum(list2))    
# print(np.sum(list3))
# f = 1
# fig, ax = plt.subplots(1,1,figsize = (7,7))
# ax = sns.swarmplot(data=(weight[-2,f][0,list3[0]],weight[-1,f][0,list2[0]]), palette=["red","blue"])
# ax = sns.pointplot(data=(weight[-2,f][0,list3[0]],weight[-1,f][0,list2[0]]), color = "black")
# ax.set_ylim([-0.9,0.9])


# stats.ks_2samp(np.abs(weight[-1,f][0,list2[0]]),np.abs(weight[-2,f][0,list3[0]]))

# list2 = (weight[-1,f] < -0.1) #* (weight[-2,f] == 0)
# list3 = (weight[-2,f] < -0.1) #* (weight[-2,f] == 0)


# list3 = (weight[-2,f] < -0.1)# * (weight[-1,f] == 0)
# ll = good_list_sep[list3[0]]
# ll = good_listRu[list2[0]]

# list2[0,:] = d_list
# list3[0,:] = d_list3

list4 = (weight[-1,f] > 0)*(-weight[-2,f] > 0)
# result = stats.linregress(weight[-1,f][list4],-weight[-2,f][list4])

# print(result.rvalue)

# 
fig, axes = plt.subplots(1,1,figsize = (6,4))

# f = 3
# y1 = np.mean(np.concatenate((Convdata[f][list2[0,:],:], Convdata[f][list3[0,:],:])),0)
# s1 = np.std(np.concatenate((Convdata[f][list2[0,:],:], Convdata[f][list3[0,:],:])),0)/np.sqrt(np.sum(list2)+np.sum(list3))

for f in [1]:
    C1 = Convdata[f][list2[0,:],:]
    C1 = np.abs(C1)
    C1 = ndimage.gaussian_filter(C1,[0,5])

    C1 = C1[np.max(np.abs(C1),1)>0,:]
    C2 = Convdata[f][list3[0,:],:]
    C2 = np.abs(C2)
    C2 = ndimage.gaussian_filter(C2,[0,5])

    C2 = C2[np.max(np.abs(C2),1)>0,:]
    
    print(np.size(C1,0))
    print(np.size(C2,0))
    # if f == 1:
    #     C1 = -C1
    #     C2 = -C2
        
        
    y1 = np.median(C1,0)
    y2 = np.median(C2,0)
    s1 = np.std(C1,0)/np.sqrt(np.size(C1,0))
    s2 = np.std(C2,0)/np.sqrt(np.size(C2,0))
    # y1 = np.mean(Convdata[f][list2[0,:],:],0)
    # s1 = np.std(Convdata[f][list2[0,:],:],0)/np.sqrt(np.sum(list2))
    # y2 = np.mean(Convdata[f][list3[0,:],:],0)
    # s2 = np.std(Convdata[f][list3[0,:],:],0)/np.sqrt(np.sum(list3))
    
    # take history units
    
    d = Convdata[f][list3[0,:],:]
    
    # p1 = np.arange(90,110)
    # t1 = Convdata[-1,f][list2[0,:],:][:,p1]
    
    # t2 = Convdata[-2,f][list3[0,:],:][:,p1]
    # stats.ks_2samp(np.mean(t1,1),np.mean(t2,1))
    
    # y1 = ndimage.gaussian_filter(y1,3)
    # y2 = ndimage.gaussian_filter(y2,3)
    
    cmap = cmap3 = ['tab:purple','tab:blue','tab:red','tab:olive']
    
    axes.plot(x_axis*1e-3-prestim*1e-3,y1,c = cmap[f],linestyle = 'solid')
    axes.fill_between(x_axis*1e-3-prestim*1e-3,y1-s1,y1+s1,facecolor = cmap[f],alpha = 0.3)
    
    
    axes.plot(x_axis*1e-3-prestim*1e-3,y2,c = cmap[f],linestyle = 'dashed')
    axes.fill_between(x_axis*1e-3-prestim*1e-3,y2-s2,y2+s2,facecolor = cmap[f],alpha = 0.3)
    
    for t in np.arange(np.size(C1,1)-10):
        [S,p] =  stats.ks_2samp(C1[:,t],C2[:,t])
        # [S,p] =  stats.ks_2samp(np.mean(C1[:,t:t+10],1),np.mean(C2[:,t:t+10],1))
        if p < 0.05:
            axes.scatter([(t*window-t_period)*1e-3],[0],marker = '*', color = 'black')
    
    
    axes.set_ylim([-0.01, 0.3])
    axes.set_xlim([-3,4])
# plt.savefig("Rew_post.svg")

t1 = 50
t2 = 80
np.median(C1[:,t1:t2])
np.median(C2[:,t1:t2])



# fig, ax = plt.subplots(1,1,figsize = (6,6))

# ax = sns.stripplot(data=(np.mean(C2[:,t1:t2],1),np.mean(C1[:,t1:t2],1)), color = "black",size=5)
# ax = sns.violinplot(data=(np.mean(C2[:,t1:t2],1),np.mean(C1[:,t1:t2],1)), 
#                     color = cmap3[f], saturation = 0.5, inner_kws=dict(box_width=15, whis_width=2, color=".3") )

# # ax = sns.pointplot(data=(np.mean(C1[:,t1:t2],1),np.mean(C2[:,t1:t2],1)), color = "black")

# ax.set_ylim([-1,1])

# stats.ks_2samp(np.mean(C1[:,t1:t2],1),np.mean(C2[:,t1:t2],1))

# %% calculatung list overlap
listOv = {}
for f in [0,3]: #np.arange(4):
    listOv[f] = (np.mean(Convdata[f],1) != 0)

list0 = listOv[0]*listOv[3]

Lg = len(good_listRu)
Lic = np.where(good_listRu <180)
Lic = Lic[0][-1]
    
# list0[Lic:Lg] = False # PPCIC
# list0[0:Lic] = False # PPCAC
# np.sum(list0)

# %% plotting weights by peak order
listOv = {}

f = 0
W5 = {}
for f in [0,3]: #np.arange(4):
    list0 = (np.mean(Convdata[f],1) != 0)
    
    Lg = len(good_listRu)
    Lic = np.where(good_listRu <180)
    Lic = Lic[0][-1]
    
    list0[Lic:Lg] = False # PPCIC
    # list0[0:Lic] = False # PPCAC
    # 
    list0ind = np.arange(Lg)
    list0ind = list0ind[list0]
    W = ndimage.uniform_filter(Convdata[f][list0,:],[0,5], mode = "mirror")
    
    max_peak = np.argmax(np.abs(W),1)
    max_ind = max_peak.argsort()
    
    list1 = []
    list2 = []
    list3 = []
    
    
    for m in np.arange(np.size(W,0)):
        n = max_ind[m]
        SD = np.std(W[n,:])
        if SD< 0.05:
            SD = 0.05
        if max_peak[n]> 20:    
            if W[n,max_peak[n]] > 3*SD:
                list1.append(m)
                list3.append(m)
            elif W[n,max_peak[n]] <-3*SD:
                list2.append(m)
                list3.append(m)
            
    max_ind1 = max_ind[list1]  
    max_ind2 = max_ind[list2]     
    max_ind3 = max_ind[list3]
    max_peak3 = max_peak[list3]
    
    listOv[f] = list0ind[list3]
    
    W1 = W[max_ind1]
    W2 = W[max_ind2]    
    W4 = np.abs(W[max_ind3])
    
    W3 = np.concatenate((W1,W2), axis = 0)
    print(np.size(W1,0))
    print(np.size(W2,0))
    fig, axes = plt.subplots(1,1,figsize = (10,10))
    
    
    clim = [-0.0, 0.5]
    im1 = axes.imshow(W4[:,20:], clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
    # im2 = axes[1].imshow(W2, clim = clim, aspect = "auto", interpolation = "None")
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im1, cax=cbar_ax)
    
    # W4IC = W4
    W5[f] = W4
    
print(np.size(np.intersect1d(listOv[0],listOv[3])))

np.save('PPCIC_Hist.npy',listOv,allow_pickle = True)
# %% plot histogram
fig, axes = plt.subplots(figsize=(6,4), nrows=1)
for f in [0,3]: #np.arange(4):
    y1 = np.mean(W5[f],0)*np.size(W5[f],0)/423
    s1 = np.std(W5[f],0)/np.sqrt(423)
    
    # y2 = np.mean(W2,0)
    # s2 = np.std(W2,0)/np.sqrt(np.size(W2,0))
    
    axes.plot(x_axis*1e-3-t_period*1e-3,y1,c = cmap[f],linestyle = 'solid')
    axes.fill_between(x_axis*1e-3-t_period*1e-3,y1-s1,y1+s1,facecolor = cmap[f],alpha = 0.3)
    axes.set_ylim([-0.01,0.15])
    axes.set_xlim([-3,4])
    # plt.savefig("PIC.svg")
 

bins = np.linspace(20,160,15)
fig, axes = plt.subplots(figsize=(6,4), nrows=1)
axes = sns.histplot(max_peak3, stat='probability',bins = bins)
# axes.hist(max_peak3,bins = bins, density = "True", stacked = "True")
axes.set_ylim([0,0.20])

# %% Analysis of Ca traces of History units
# Compare Hit vs Miss, FA vs CR 
f  = 0
Yc = {}
t = 0
Yc[f] = {}
Rate = {}
for tt in np.arange(6):
    Yc[f][tt] = np.zeros((np.size(listOv[f]),140))
    Rate[tt] = np.zeros((np.size(listOv[f]),1))
# Yc[f][1] = np.zeros((np.size(listOv[f]),140))
for n in listOv[f]:
    l ={};
    nn = int(good_listRu[n])
    Y = Data[nn, c_ind-1]["Y"]
    X = D_ppc[nn,2][:,2:6]
    c1 = 200
    c2 = D_ppc[nn,4][0][0] + 25
    # Y = Y[c1:c2]
    X = X[c1:c2]
    Xpre = np.concatenate(([[1,1,1,0]],X[0:-1,:]),0)
    Xpre = np.column_stack((Xpre[:,:-1],Xpre[:,1]*Xpre[:,2]))
    # l[0] = ((X[:,0]==0)*(X[:,1]==1))*(Xpre[:,3]==1) # Rw pre FA post
    # l[1] = ((X[:,0]==0)*(X[:,1]==1)) # FA post
    # l[2] = ((X[:,0]==0)*(X[:,1]==0))*(Xpre[:,3]==1) # Rw pre CR post
    # l[3] = ((X[:,0]==0)*(X[:,1]==0))# 
    
    l[0] = ((X[:,2]==0)*(Xpre[:,2]==0))# pre incorrect post incorrect
    l[1] = ((X[:,2]==0)*(Xpre[:,2]==1))# pre correct post incorrect
    l[2] = ((X[:,2]==1)*(Xpre[:,2]==0))# pre incorrect post correct
    l[3] = ((X[:,2]==1)*(Xpre[:,2]==1))# pre correct post correct
    
    # l[0] = ((X[:,0]==1)*(X[:,1]==1))*(Xpre[:,3]==1) # Rw pre FA post
    # l[1] = ((X[:,0]==1)*(X[:,1]==1))*(Xpre[:,3]==0) # nRw pre FA post
    # l[2] = ((X[:,0]==1)*(X[:,1]==0))*(Xpre[:,3]==1) # Rw pre CR post
    # l[3] = ((X[:,0]==1)*(X[:,1]==0))*(Xpre[:,3]==0) # 
    
    # l[0] = ((X[:,1]==1)) # Rw pre lick post
    # l[1] = ((X[:,1]==1))*(Xpre[:,3]==0) # nRw pre lick post
    # l[2] = ((X[:,1]==0)) # Rw pre no_lick post
    # l[3] = ((X[:,1]==0))*(Xpre[:,3]==0) # 
    
    for tt in np.arange(4):
        Yc[f][tt][t,:] = np.mean(Y[l[tt],20:],0)/(np.max((np.mean(Y[:,20:],0))+0.5))
        Rate[tt][t,0] = np.sum(l[tt])#/np.size(l[tt],0)
    
    if np.sum(l[0]) >0:
        Yc[f][4][t,:] = (np.mean(Y[l[0],20:],0) + np.mean(Y[l[1],20:],0))/(2*(np.max((np.mean(Y[:,20:],0))+0.5)))
    else:
        Yc[f][4][t,:] = Yc[f][1][t,:]
        
    Yc[f][5][t,:] = (np.mean(Y[l[2],20:],0) + np.mean(Y[l[3],20:],0))/(2*(np.max((np.mean(Y[:,20:],0))+0.5)))
    # Yc[f][1][t,:] = np.mean(Y[((X[:,0]==0)*(X[:,1]==1)),20:],0)/(np.max((np.mean(Y[:,20:],0))+0.5))
    # Yc[f][0][t,:] = np.mean(Y[((X[:,0]==0)*(X[:,1]==0)),20:],0)/(np.max((np.mean(Y[:,20:],0))+0.5)) 
    # Yc[f][1][t,:] = np.mean(Y[((X[:,1]==1)),20:],0)/(np.max((np.mean(Y[:,20:],0))+0.5))
    # Yc[f][0][t,:] = np.mean(Y[((X[:,1]==0)),20:],0)/(np.max((np.mean(Y[:,20:],0))+0.5))
     
    t += 1


cmap4 = ['tab:red','tab:red','tab:green','tab:green','red','green']
fig, axes = plt.subplots(1,1,figsize = (10,8))
lc = ['dotted','solid','dotted','solid','solid','solid']
for tt in [1,3]:
    y1 = np.nanmean(Yc[f][tt],0)
    y1 = ndimage.gaussian_filter(y1,2)
    s1 = np.nanstd(Yc[f][tt],0)/np.sqrt(np.size(listOv[f]))
    axes.plot(x_axis[20:]*1e-3-t_period*1e-3,y1,c = cmap4[tt],linestyle = lc[tt])
    axes.fill_between(x_axis[20:]*1e-3-t_period*1e-3,y1-s1,y1+s1,facecolor = cmap4[tt],alpha = 0.3)


# %% building SVM with Ca data 

# using 4FA and 4CR trials for training and equal number for testing
ntr1 = 5
ntr2 = 5
f = 0

Xm = np.zeros((np.size(listOv[f]),ntr1*2))
Xt = np.zeros((np.size(listOv[f]),ntr2*2))
cv = 20
Acc = np.zeros ((cv,160))
Accshuffle= np.zeros ((cv,160))
for t in np.arange(20,160):
    if np.mod(t,20) == 0:
        print("print :  {} /160" .format(t))
    for k in np.arange(cv):
        tt = 0
        for n in listOv[f]:
            nn = int(good_listRu[n])
            Y = Data[nn, c_ind-1]["Y"]
            X = D_ppc[nn,2][:,2:6]
            c1 = 200
            c2 = D_ppc[nn,4][0][0] + 25
            # Y = Y[c1:c2]
            X = X[c1:c2]
            Xpre = np.concatenate(([[1,1,1,0]],X[0:-1,:]),0)
            Xpre = np.column_stack((Xpre[:,:-1],Xpre[:,1]*Xpre[:,2]))
            # l[0] = np.argwhere(((X[:,0]==0)*(X[:,1]==0))) # CR
            # l[1] = np.argwhere(((X[:,0]==0)*(X[:,1]==1))) # FA
            l[0] = np.argwhere((X[:,2]==0)) # incorrect
            l[1] = np.argwhere((X[:,2]==1)) # correct
            if np.size(l[0],0)>=ntr1+ntr2 and np.size(l[1],0)>=ntr1+ntr2:
                b = np.random.choice(l[0].ravel(),ntr1,replace = False)
                c = np.random.choice(np.setdiff1d(l[0].ravel(),b),ntr2,replace = False)
                Xm[tt,0:ntr1] = Y[b,t]
                Xt[tt,0:ntr2] = Y[c,t]
                b = np.random.choice(l[1].ravel(),ntr1,replace = False)
                c = np.random.choice(np.setdiff1d(l[1].ravel(),b),ntr2,replace = False)
                Xm[tt,ntr1:ntr1*2] = Y[b,t]
                Xt[tt,ntr2:ntr2*2] = Y[c,t]
                
                tt += 1
                    
        Ym = [0,0,0,0,0,1,1,1,1,1]
        
        
        # Define the model
        log_reg = LogisticRegression(penalty='none')
        
        # Fit it to data
        log_reg.fit(Xm.T, Ym)
        
        Yhat = log_reg.predict(Xt.T)
        Ym2 = [0,0,0,0,0,1,1,1,1,1]
        Acc[k,t] = 1-np.sum(np.abs(Yhat-Ym2))/(ntr2*2)
        np.random.shuffle(Xt)
        Yhat = log_reg.predict(Xt.T)
        Accshuffle[k,t] = 1-np.sum(np.abs(Yhat-Ym2))/(ntr2*2)


y1 = np.mean(ndimage.gaussian_filter(Acc,[0,2]),0)
s1 = np.std(ndimage.gaussian_filter(Acc,[0,2]),0)/2# /np.sqrt(cv)

y2 = np.mean(ndimage.gaussian_filter(Accshuffle,[0,2]),0)
s2 = np.std(ndimage.gaussian_filter(Accshuffle,[0,2]),0)/2 #/np.sqrt(cv)   #/np.sqrt(8)
fig, axes = plt.subplots(1,1,figsize = (10,8))
axes.plot(x_axis*1e-3-t_period*1e-3,y1,c = 'blue')
axes.fill_between(x_axis*1e-3-t_period*1e-3,y1-s1,y1+s1,facecolor = 'blue',alpha = 0.3)
axes.plot(x_axis*1e-3-t_period*1e-3,y2,c = 'black')
axes.fill_between(x_axis*1e-3-t_period*1e-3,y2-s2,y2+s2,facecolor = 'black',alpha = 0.3)

axes.set_xlim([-2.8,4])
axes.set_ylim([0.4,1])


# %% 

stats.ks_2samp(np.mean(W4AC[:,100:140],1), np.mean(W4IC[:,100:140],1))


a_mean = [np.mean(W4AC[:,20:60]),np.mean(W4AC[:,60:100]),np.mean(W4AC[:,100:140])]
i_mean = [np.mean(W4IC[:,20:60]),np.mean(W4IC[:,60:100]),np.mean(W4IC[:,100:140])]

a_std = [np.std(W4AC[:,20:60]),np.std(W4AC[:,60:100]),np.std(W4AC[:,100:140])]/np.sqrt(np.size(W4AC,0))
i_std = [np.std(W4IC[:,20:60]),np.std(W4IC[:,60:100]),np.std(W4IC[:,100:140])]/np.sqrt(np.size(W4IC,0)) 

fig, axes = plt.subplots(1,1,figsize = (10,8))
axes.bar(np.arange(3)*2+0.7,i_mean,yerr = i_std, color = 'tab:olive', alpha = 0.5, width = 0.5)
axes.bar(np.arange(3)*2,a_mean, yerr= a_std, color = 'tab:olive', alpha = 1, width = 0.5)
# axes.set_ylim([0,0.8])
 
# %%
fig, axes = plt.subplots(figsize=(7,4), nrows=1)

y1 = np.mean(W4,0)
s1 = np.std(W4,0)/np.sqrt(np.size(W4,0))

y2 = np.mean(W2,0)
s2 = np.std(W2,0)/np.sqrt(np.size(W2,0))

axes.plot(x_axis*1e-3-t_period*1e-3,y1,c = cmap[f],linestyle = 'solid')
axes.fill_between(x_axis*1e-3-t_period*1e-3,y1-s1,y1+s1,facecolor = cmap[f],alpha = 0.3)
axes.set_ylim([-0.01,0.12])
      
# axes[1].plot(x_axis*1e-3-t_period*1e-3,y2,c = cmap[f],linestyle = 'solid')
# axes[1].fill_between(x_axis*1e-3-t_period*1e-3,y2-s2,y2+s2,facecolor = cmap[f],alpha = 0.3)
# axes[1].set_ylim([-0.2,0.01])

fig.tight_layout()  
# plt.savefig("RH_PAC.svg")


# list3_AC = list3
# list3_IC = list3
# %%bar plot 
cmap3 = ['tab:blue','tab:red','tab:olive']
b2 = [187, 115, 268]
b2 = np.array(b2)/403

b1 = [43, 37, 71]
b1 = np.array(b1)/129


fig, axes = plt.subplots(1,1,figsize = (10,8))
axes.bar(np.arange(3)*2,b1, color = cmap3, alpha = 0.5, width = 0.5)
axes.bar(np.arange(3)*2+0.7,b2, color = cmap3, alpha = 1, width = 0.5)
axes.set_ylim([0,0.8])


# %%


fig, axes = plt.subplots(1,1,figsize = (8,8))



sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
fig, ax = plt.subplots(1,1,figsize = (7,7))

# ax = sns.boxplot(data=(W1a,W3a,W2a,W4a), showfliers = False)
ax = sns.stripplot(data=(max_peak1, max_peak2), palette=["red","blue"])
# ax = sns.pointplot(data=(W1a,W 3a,W2a,W4a), color="black", join = False)

# ax.set_ylim([-0.9,0.9])

plt.show()

stats.ks_2samp(max_peak1, max_peak2)


# %% calculate number of positive vs negative weights per time period. 

weight = {}
p = {}
# p[-1] = np.arange(20,50)
pe21 = []
pe22 = []
pe31 = []
pe32 = []
Convmean = {}
Convstd  = {}
for c_ind in [-1,-2]:
    Convmean[c_ind] = np.zeros((2,15))
    Convstd[c_ind] = np.zeros((2,15))

Lw = [90,80,70,86]
# Lw = [238,274,218,205]



nf = 4
thresh = 0.1
Lg = np.size(Convdata[f],0)

for f in [1]: #np.arange(nf):
    pe21 = []
    pe22 = []
    t = 0
    fig, axes  = plt.subplots(figsize=(10,10))
    for w in np.arange(15)*10:
        p[-1] = np.arange(w,w+20)
        
        p[-2] = p[-1]
    
        weight[-1,f]= np.zeros((1,Lg))
        for n in np.arange(Lg):
            c_ind = -1
            weight[c_ind,f][0,n] = np.mean(Convdata[f][n,p[c_ind]])
        
        
        list2 = (np.abs(weight[c_ind,f]) > thresh) #* (weight[-2,f] == 0)    
        list21 = (weight[c_ind,f] > thresh) #* (weight[-2,f] == 0)
        list22 = (weight[c_ind,f] < -thresh)
        axes.scatter(w*np.ones((1,np.sum(list2))),weight[c_ind,f][list2],c = "black") #cmap3[f])

            
        xticks = np.linspace(0,140,8)
        xlabels = np.linspace(-3.5,3.5,8)
        xlabels = [str(x) for x in xlabels]
        axes.set_xticks(xticks, xlabels)
        
        Convmean[c_ind][0,t] = np.mean(weight[c_ind,f][list21])
        Convmean[c_ind][1,t] = np.mean(weight[c_ind,f][list22])
        Convstd[c_ind][0,t] = np.std(weight[c_ind,f][list21])/np.sqrt(np.sum(list21))
        Convstd[c_ind][1,t] = np.std(weight[c_ind,f][list22])/np.sqrt(np.sum(list22))
        
        # test1 = ndimage.gaussian_filter(Convdata[f][list21[0,:],:],[0,3])
        # test2 = ndimage.gaussian_filter(Convdata[f][list22[0,:],:],[0,3])
        
        # test1 = Convdata[f][list21[0,:],:]
        # test2 = Convdata[f][list22[0,:],:]
            
        # Convmean[c_ind][0,t] = np.mean(test1[:,p[c_ind]])
        # Convmean[c_ind][1,t] = np.mean(test2[:,p[c_ind]])
        # Convstd[c_ind][0,t] = np.std(test1[:,p[c_ind]])/np.sqrt(np.sum(list21))
        # Convstd[c_ind][1,t] = np.std(test2[:,p[c_ind]])/np.sqrt(np.sum(list22))
        t += 1    
            
        pe21.append(np.sum(list21))
        pe22.append(np.sum(list22))
        
    
    index = np.linspace(0,140,15)
    axes.plot(index,Convmean[c_ind][0,:]*2,linewidth = 4)
    axes.plot(index,Convmean[c_ind][1,:]*2,linewidth = 4)
    axes.set_ylim([-0.9,0.9])
    axes.hlines(0,0,160,color = 'black')
    
    index = np.linspace(-3.5,3.5,15)
    fig, axes = plt.subplots(figsize=(5,5), nrows=2, sharex=True)
    fig.tight_layout()
    axes[0].bar(index, np.array(pe21)/Lw[f], align='center', color=cmap[f],width = 0.4)
    axes[1].bar(index, -np.array(pe22)/Lw[f], align='center', color=cmap[f],width = 0.4,alpha = 0.7)
    ythresh = 1
    axes[1].set_ylim([-ythresh,0])
    axes[0].set_ylim([0,ythresh])



# %% calculate number of positive vs negative weights per time period. 

weight = {}
p = {}
# p[-1] = np.arange(20,50)
pe21 = []
pe22 = []
pe31 = []
pe32 = []

f = 0

fig, axes = plt.subplots(figsize=(15,7), nrows = 1, ncols = 2, sharex = True )

for w in np.arange(15)*10:
    p[-1] = np.arange(w,w+20)
    
    p[-2] = p[-1]
    # p[-2] = np.arange(0,15)
    
    Lg = len(good_listRu)
    Lic = 151     
    Lac = Lg-Lic
    weight[-1,f]= np.zeros((1,Lg))
    weight[-2,f] = np.zeros((1,Lg))    
    for c_ind in [-1,-2]:
        if c_ind == -1:
            for n in np.arange(Lic,Lg):
                weight[c_ind,f][0,n] = np.mean(Convdata[f][n,p[c_ind]])
        if c_ind == -2:
            for n in np.arange(Lic):
                weight[c_ind,f][0,n] = np.mean(Convdata[f][n,p[c_ind]])
    
    
    # list21 = (weight[-1,f] > 0.1) #* (weight[-2,f] == 0)
    list21 = (np.abs(weight[-1,f]) > 0)
    list22 = (weight[-1,f] < -0.1)
    
    axes[0].scatter(w*np.ones((1,np.sum(list21))),weight[-1,f][list21],c = "black",s = 25, alpha = 0.3) #cmap3[f])
    # axes[1,0].scatter(w*np.ones((1,np.sum(list22))),weight[-1,f][list22],c = "black",s = 25, alpha = 0.3) #cmap3[f])
    
    
    
    # list31 = (weight[-2,f] > 0.1)# * (weight[-1,f] == 0)
    list31 = (np.abs(weight[-2,f]) > 0)
    list32 = (weight[-2,f] < -0.1)# * (weight[-1,f] == 0)
    axes[1].scatter(w*np.ones((1,np.sum(list31))),weight[-2,f][list31],c = "black",s = 25, alpha = 0.2)
    pe21.append(np.sum(list21))
    pe22.append(np.sum(list22))
    pe31.append(np.sum(list31))
    pe32.append(np.sum(list32))


xticks = np.linspace(0,140,8)
xlabels = np.linspace(-1.5,5.5,8)
xlabels = [str(x) for x in xlabels]
for a in [0,1]:
    axes[a].set_ylim([-0.9,0.9])
    axes[a].hlines(0,0,160,color = 'black')
    axes[a].set_xticks(xticks, xlabels)

index = np.linspace(-1.5,5.5,15)

# fig, axes = plt.subplots(figsize=(5,5), nrows=2, sharex=True)
# fig.tight_layout()

# Lfic = [103,98,80,86]
# Lfac = [238,274,218,205]

# # Lw = [103,98,80,86]
# # Lw = [238,274,218,205]

# axes[0].bar(index, np.array(pe21)/Lfac[f], align='center', color=cmap[f],width = 0.3)
# axes[1].bar(index, -np.array(pe22)/Lfac[f], align='center', color=cmap[f],width = 0.3,alpha = 0.7)
# axes[1].set_ylim([-0.4,0])
# axes[0].set_ylim([0,0.4])


# fig, axes = plt.subplots(figsize=(5,5), nrows=2, sharex=True)
# fig.tight_layout()

# axes[0].bar(index, np.array(pe31)/Lfic[f], align='center', color=cmap[f],width = 0.3)
# axes[1].bar(index, -np.array(pe32)/Lfic[f], align='center', color=cmap[f],width = 0.3,alpha = 0.7)
# axes[1].set_ylim([-0.4,0])
# axes[0].set_ylim([0,0.4])
# %% 
weight = {}
nbunits = {}
for f in np.arange(ax_sz): 
    nbunits[-1,f] = np.zeros((1,7))
    nbunits[-2,f] = np.zeros((1,7))

for t in np.arange(7):
    p = np.arange(t*20,t*20+20)
    for f in np.arange(ax_sz):  
        weight[-1,f]= np.zeros((1,Lg))
        weight[-2,f] = np.zeros((1,Lg))

        for c_ind in [-1,-2]:
            if c_ind == -1:
                for n in np.arange(Lic,Lg):
                    weight[c_ind,f][0,n] = np.mean(Convdata[f][n,p])
                nbunits[c_ind,f][0,t] = np.sum((weight[c_ind,f] > 0.1))/Lac
    
            if c_ind == -2:
                for n in np.arange(Lic):
                    weight[c_ind,f][0,n] = np.mean(Convdata[f][n,p])
                nbunits[c_ind,f][0,t] = np.sum((weight[c_ind,f] > 0.1))/Lic



fig, axes = plt.subplots(4,1, figsize = (8,20))
for f in np.arange(ax_sz):
    axes[f].plot(np.arange(7), nbunits[-1,f].T, c = cmap[f], linestyle = "solid")
    axes[f].plot(np.arange(7), nbunits[-2,f].T, c = cmap[f], linestyle = "dotted")
    

   
array = np.load('full_words.npy',allow_pickle= True)