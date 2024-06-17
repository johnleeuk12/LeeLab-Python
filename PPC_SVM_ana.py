# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:50:23 2023

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
from sklearn.linear_model import TweedieRegressor, Ridge, ElasticNet, Lasso,LogisticRegression
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
    # Ca trace bin is approx 50ms
    Ca_window = 50
    Y1 = np.zeros((N_trial,int(t_period/Ca_window)))
    Y = np.zeros((N_trial,int(t_period/window)))
    for tr in range(N_trial):
        Y1[tr,:] = Yraw[0,D_ppc[n,2][tr,0]-1 - int(prestim/Ca_window): D_ppc[n,2][tr,0] + int(t_period/Ca_window)-1 - int(prestim/Ca_window)]
    
    if Ca_window == window:
        Y = Y1
    else:
        for w in np.arange(int(t_period/window)):
            Y[:,w] = (Y1[:,2*w]+Y1[:,2*w+1])/2
        
    # Yraw3 = Yraw-Ymed_wind+np.mean(Yraw)
    
    # Original Y calculation #####
    
    # Y = np.zeros((N_trial,int(t_period/window)))
    # for tr in range(N_trial):
    #     Y[tr,:] = Yraw[0,D_ppc[n,2][tr,0]-1 - int(prestim/window): D_ppc[n,2][tr,0] + int(t_period/window)-1 - int(prestim/window)]

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
        c2 = D_ppc[n,4][0][0] + 50
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
        c1 = D_ppc[n,4][0][0] + 25
        Y = Y[c1:]
        X = X[c1:]
        L2 = L2[c1:]
        

    
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

    # X2 = np.column_stack([Xprestim,-1*X[:,3]+1,
    #                       X[:,2]*X[:,1],Xpre]) 
    
    XHit = (X[:,0]==1)*(X[:,1]==1)
    XFA  = (X[:,0]==0)*(X[:,1]==1)
    Xmiss = (X[:,0]==1)*(X[:,1]==0)
    XCR = (X[:,0]==0)*(X[:,1]==0)
    
    X2 = np.column_stack([XHit,Xmiss,XFA,XCR])
    X2 = np.row_stack([[0,0,0,0],X2[:-1,:]]) # previous hit FA miss etc
    X2 = np.column_stack([X2,XHit,Xmiss,XFA,XCR])
    
    # Adding Lick(action as well)
    # X2 = np.column_stack([Xprestim,-1*X[:,3]+1,X[:,1],
    #                       X[:,2]*X[:,1],Xpre]) 


    
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
    # t_period = t_period + 3500
    t_period = t_period + prestim
    Yhat = [];
    # Yhat1 = [];
    # Yhat2 = [];
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
            

        # adding kernels to each task variable
        # if w*window <= 1500-4*window:
        #     X3[:,1:] = 0;
        # elif w*window <= 8000-4*window:
        #     X3[:,1:3]= 0;
        # elif w*window <= 8000+1500-4*window:
        #     X3[:,2] = 0;
            
        # if w*window > 8000-4*window:
        #     X3[:,0] = 0;
        #     X3[:,3] = 0;

                        
        
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
            cmap = ['tab:purple','tab:blue','tab:red','tab:orange']
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
        # stim_ind1 = X[:,1] == 1     
        # stim_ind2 = X[:,1] == 0  
        
    
        # ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind1,:],0),2),
        #           linewidth = 2.0, color = cmap[2],label = 'Go',linestyle = lstyles[3])
        # ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind2,:],0),2),
        #           linewidth = 2.0, color = cmap[2],label = 'NoGo',linestyle = 'dashed')
        # ax1.set_title('Firing rate y')
        # ax1.legend(loc = 'upper right')
    
        
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

# %% Run main GLM code
"""     
Each column of X contains the following information:
    0 : contingency 
    1 : lick vs no lick
    2 : correct vs wrong
    3 : stim 1 vs stim 2
    4 : if exists, would be correct history (previous correct ) 

"""



t_period = 6500
prestim = 1500

window = 100 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 10 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [3]
c_ind = 3


if ca ==0:
    D_ppc = load_matfile()
    # good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    good_list = find_good_data_Ca(t_period)

    
PAC_list = np.load('PPC_Hist.npy',allow_pickle= True).item()

ind =1 # 0 for IC 1 for AC
f = 0
# PAC_list = PAC_list[ind,f]

x_axis = np.arange(1, t_period+prestim, window)
Acc_all = {}

# %% Analysis of Ca traces of History units
# Compare Hit vs Miss, FA vs CR 
# f  = 0
Yc = {}
t = 0
ind =1 # 0 for IC 1 for AC
f = 0
c_ind =3
# PAC_
Yc[f] = {}
Rate = {}
ylens = int((t_period+prestim)/window)
good_list_sep = np.arange(np.size(D_ppc,0))
PAC_list[0,0] = good_list_sep[good_list_sep<180] 
PAC_list[1,0] = good_list_sep[good_list_sep>180] 

# PAC_list[0,0] = good_list[good_list<180] 
# PAC_list[1,0] = good_list[good_list>180] 
for tt in np.arange(6):
    Yc[f][tt] = np.zeros((np.size(PAC_list[ind,f]),ylens))
    Rate[tt] = np.zeros((np.size(PAC_list[ind,f]),1))
# Yc[f][1] = np.zeros((np.size(listOv[f]),140))
for n in PAC_list[ind,f]:
    n = int(n)
    X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
    
    for tt in np.arange(4):
        Yc[f][tt][t,:] = np.mean(Y[(X[:,tt]==1),:],0)/(np.max((np.mean(Y[:,:],0))+0.5))
        Rate[tt][t,0] = np.sum(X[:,tt])#/np.size(l[tt],0)
    
    # if np.sum(l[0]) >0:
    #     Yc[f][4][t,:] = (np.mean(Y[l[0],20:],0) + np.mean(Y[l[1],20:],0))/(2*(np.max((np.mean(Y[:,20:],0))+0.5)))
    # else:
    #     Yc[f][4][t,:] = Yc[f][1][t,:]
        
    # Yc[f][5][t,:] = (np.mean(Y[l[2],20:],0) + np.mean(Y[l[3],20:],0))/(2*(np.max((np.mean(Y[:,20:],0))+0.5)))
    # Yc[f][1][t,:] = np.mean(Y[((X[:,0]==0)*(X[:,1]==1)),20:],0)/(np.max((np.mean(Y[:,20:],0))+0.5))
    # Yc[f][0][t,:] = np.mean(Y[((X[:,0]==0)*(X[:,1]==0)),20:],0)/(np.max((np.mean(Y[:,20:],0))+0.5)) 
    # Yc[f][1][t,:] = np.mean(Y[((X[:,1]==1)),20:],0)/(np.max((np.mean(Y[:,20:],0))+0.5))
    # Yc[f][0][t,:] = np.mean(Y[((X[:,1]==0)),20:],0)/(np.max((np.mean(Y[:,20:],0))+0.5))
     
    t += 1


# cmap4 = ['tab:red','tab:red','tab:green','tab:green','red','green']

cmap4 =  ['tab:blue','tab:gray','tab:red','tab:green']

fig, axes = plt.subplots(1,1,figsize = (10,8))
lc = ['dotted','solid','dotted','solid','solid','solid']
for tt in [0,2,3]: #np.arange(4): #[1,3]:
    y1 = np.nanmean(Yc[f][tt],0)
    y1 = ndimage.gaussian_filter(y1,2)
    s1 = np.nanstd(Yc[f][tt],0)/np.sqrt(np.size(PAC_list[ind,f]))
    axes.plot(x_axis,y1,c = cmap4[tt],linestyle = lc[tt])
    axes.fill_between(x_axis,y1-s1,y1+s1,facecolor = cmap4[tt],alpha = 0.3)


# %% building SVM with Ca data 

# using 4FA and 4CR trials for training and equal number for testing
ntr1 = 15 # train
ntr2 = 15 # test
# ind = 0
# c_ind = 2

f = 0
ylens = len(y1)
Xm = np.zeros((np.size(PAC_list[ind,f]),ntr1*2))
Xt = np.zeros((np.size(PAC_list[ind,f]),ntr2*2))
cv = 40
Acc = np.zeros ((cv,ylens))
Accshuffle= np.zeros ((cv,ylens))

ModelW = {}
for k in np.arange(cv):
    ModelW[k] = np.zeros((np.size(PAC_list[ind,f]),ylens))
# from sklearn.model_selection import train_test_split
from sklearn import svm

t_ind = 2 # Hit

Data = {}
for n in PAC_list[ind,f]:
    n = int(n)
    X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
    l = {}
    # l[0] = np.concatenate([np.argwhere((X[:,3]==1)),np.argwhere((X[:,0]==1))]) # Current Correct(Check if previous)
    # l[1] = np.concatenate([np.argwhere((X[:,1]==1)),np.argwhere((X[:,2]==1))])# Currect IC
    l[0] = np.argwhere((X[:,0]==1)) #np.concatenate([np.argwhere((X[:,0]==1)),np.argwhere((X[:,3]==1))]) # Current Correct(Check if previous)
    l[1] = np.argwhere((X[:,2]==1)) #np.concatenate([np.argwhere((X[:,1]==1)),np.argwhere((X[:,2]==1))])# Currect IC
    Data[n] = {"X":X,"l" : l,"Y":Y}


for t in np.arange(ylens):
    if np.mod(t,5) == 0:
        print("print :  {} /80" .format(t))
        
    
    for k in np.arange(cv):
        tt = 0
        Xm = np.zeros((np.size(PAC_list[ind,f]),ntr1*2))
        Xt = np.zeros((np.size(PAC_list[ind,f]),ntr2*2))
        for n in PAC_list[ind,f]:
            X = Data[n]["X"]
            l = Data[n]["l"]
            Y = Data[n]["Y"]
            # if np.size(l[1],0) > ntr1 and np.size(l[0],0) > ntr1:
            if np.size(l[0],0)>=ntr1+ntr2 and np.size(l[1],0)>=ntr1+ntr2:
                b = np.random.choice(l[0].ravel(),ntr1,replace = False)
                c = np.random.choice(np.setdiff1d(l[0].ravel(),b),ntr2,replace = False)    
                Xm[tt,0:ntr1] = Y[b,t]-np.median(Y[:,:])
                Xt[tt,0:ntr2] = Y[c,t]-np.median(Y[:,:])
                    
                b = np.random.choice(l[1].ravel(),ntr1,replace = False)
                c = np.random.choice(np.setdiff1d(l[1].ravel(),b),ntr2,replace = False)                
                Xm[tt,ntr1:ntr1*2] = Y[b,t]-np.median(Y[:,:])
                Xt[tt,ntr2:ntr2*2] = Y[c,t]-np.median(Y[:,:])
            tt += 1

        Ym = np.concatenate((np.zeros((ntr1,1)),np.ones((ntr1,1)))) #[0,0,0,0,0,1,1,1,1,1]
        # Ym = np.concatenate((np.ones((ntr1,1))*-1,np.ones((ntr1,1)))) #[0,0,0,0,0,1,1,1,1,1]

        Ym = Ym.ravel()
        Ym2 = np.concatenate((np.zeros((ntr2,1)),np.ones((ntr2,1))))
        # Ym2 = np.concatenate((np.ones((ntr1,1))*-1,np.ones((ntr2,1))))

        Ym2 = Ym2.ravel()
        # Define the model
        # log_reg = LogisticRegression(penalty='none')
        log_reg = svm.SVC(kernel='linear', C=0.1)        
        # Xm[[0,10],:] = 0
        # Fit it to data
        clf = log_reg.fit(Xm.T, Ym)
        ModelW[k][:,t] = clf.coef_
        Acc[k,t] = clf.score(Xt.T, Ym2)
        # coef = log_reg.coef_
        # Yhat = log_reg.predict(Xt.T)
        # Ym2 = np.concatenate((np.zeros((ntr1,1)),np.ones((ntr2,1))))
        # Ym2 = Ym2.ravel()
        # Acc[k,t] = 1-np.sum(np.abs(Yhat-Ym2))/(ntr2*2)
        # # print(Acc[k,t])
        np.random.shuffle(Xt)
        Yhat = log_reg.predict(Xt.T)
        Accshuffle[k,t] = 1-np.sum(np.abs(Yhat-Ym2))/(ntr2*2)


y1 = np.mean(ndimage.gaussian_filter(Acc,[0,1]),0)
s1 = np.std(ndimage.gaussian_filter(Acc,[0,1]),0)# /np.sqrt(cv)

y2 = np.mean(ndimage.gaussian_filter(Accshuffle,[0,1]),0)
s2 = np.std(ndimage.gaussian_filter(Accshuffle,[0,1]),0)#/np.sqrt(cv)   #/np.sqrt(8)
fig, axes = plt.subplots(1,1,figsize = (10,8))
axes.plot(x_axis*1e-3-prestim*1e-3,y1,c = 'blue')
axes.fill_between(x_axis*1e-3-prestim*1e-3,y1-s1,y1+s1,facecolor = 'blue',alpha = 0.3)
axes.plot(x_axis*1e-3-prestim*1e-3,y2,c = 'black')
axes.fill_between(x_axis*1e-3-prestim*1e-3,y2-s2,y2+s2,facecolor = 'black',alpha = 0.3)
axes.set_ylim([0.45,1])
# axes.set_xlim([-2.8,4])
# axes.set_ylim([0.4,1])
Acc_all[t_ind] = Acc



# %%


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

Data2 = {}
ntr3 = 5

Xt2 = np.zeros((np.size(PAC_list[ind,f]),ntr3*2))
Xp1 = np.zeros((np.size(PAC_list[ind,f]),ylens))
Xp2 = np.zeros((np.size(PAC_list[ind,f]),ylens))
Yp1 = np.zeros((cv,ylens))
Yp2 = np.zeros((cv,ylens))
ModelWmean = np.zeros((np.size(PAC_list[ind,f]),ylens))
decoder = {}
t1 = 15
t2 = 30

pre_t1 = 10
# pre_t2


for k in np.arange(cv):
    ModelWmean = ModelWmean + ModelW[k]
    decoder[k]= np.mean(ModelW[k][:,t1:t2],1)

ModelWmean = ModelWmean/cv


# decoder = np.mean(ModelWmean[:,t1:t2],1)
Acc2 = np.zeros ((cv,ylens))

for n in PAC_list[ind,f]:
    n = int(n)
    X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
    l = {}
    # l[0] = np.concatenate([np.argwhere((X[:,3]==1)),np.argwhere((X[:,0]==1))]) # Current Correct(Check if previous)
    # l[1] = np.concatenate([np.argwhere((X[:,1]==1)),np.argwhere((X[:,2]==1))])# Currect IC
    l[0] = np.argwhere((X[:,4]==1)) #np.concatenate([np.argwhere((X[:,0]==1)),np.argwhere((X[:,3]==1))]) # Current Correct(Check if previous)
    l[1] = np.argwhere((X[:,6]==1)) #np.concatenate([np.argwhere((X[:,1]==1)),np.argwhere((X[:,2]==1))])# Currect IC
    Data2[n] = {"X":X,"l" : l,"Y":Y}


for t in np.arange(ylens):
    if np.mod(t,5) == 0:
        print("print :  {} /80" .format(t))
    for k in np.arange(cv):
        tt = 0
        # Xt2 = np.zeros((np.size(PAC_list[ind,f]),ntr3*2))
        for n in PAC_list[ind,f]:
            X = Data2[n]["X"]
            l = Data2[n]["l"]
            Y = Data2[n]["Y"]
            if np.size(l[0],0)>=ntr3 and np.size(l[1],0)>=ntr3:
                Xp1[tt,t] = np.mean(Y[l[0],t],0) #- np.median(Y[:,:]) # trial averaged neural response for condition 1
                Xp2[tt,t] = np.mean(Y[l[1],t],0) #- np.median(Y[:,:])
                
                # b = np.random.choice(l[0].ravel(),ntr3,replace = False)
                # Xt2[tt,0:ntr3] = Y[b,t]
                    
                # c = np.random.choice(l[1].ravel(),ntr3,replace = False)
                # Xt2[tt,ntr3:ntr3*2] = Y[c,t]
            tt += 1
        # Ym = np.concatenate((np.zeros((ntr3,1)),np.ones((ntr3,1)))) #[0,0,0,0,0,1,1,1,1,1]
        # Ym = Ym.ravel()
        # Yhat = decoder.T @ Xt2
        # # Yhat = sigmoid(Yhat)
        # Yhat = (Yhat>0)*1
        Yp1[k,t] = Xp1[:,t] @ decoder[k]
        Yp2[k,t] = Xp2[:,t] @ decoder[k]
        # Acc2[k,t] = 1-np.sum(np.abs(Yhat-Ym))/(ntr3)


        
            
        
y1 = np.mean(ndimage.gaussian_filter(Yp1,[0,1]),0)
s1 = np.std(ndimage.gaussian_filter(Yp1,[0,1]),0)# /np.sqrt(cv)

y2 = np.mean(ndimage.gaussian_filter(Yp2,[0,1]),0)
s2 = np.std(ndimage.gaussian_filter(Yp2,[0,1]),0)# /np.sqrt(cv)
        
        
fig, axes = plt.subplots(1,1,figsize = (10,8))
axes.plot(x_axis*1e-3-prestim*1e-3,y1,c = 'blue')
axes.fill_between(x_axis*1e-3-prestim*1e-3,y1-s1,y1+s1,facecolor = 'blue',alpha = 0.3)     

axes.plot(x_axis*1e-3-prestim*1e-3,y2,c = 'red')
axes.fill_between(x_axis*1e-3-prestim*1e-3,y2-s2,y2+s2,facecolor = 'red',alpha = 0.3)    
   
# axes.set_ylim([0.45,1])
        
        
        
# %% calculate percentage of next FA or Hit influenced by previous FA or Hit        
        
        
pHit = [0,0,0,0];
pFA = [0,0,0,0];
for n in PAC_list[ind,f]:
    n = int(n)
    X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
    l[0] = np.argwhere((X[:,0]==1)) 
    l[1] = np.argwhere((X[:,2]==1)) 
    XHit = X[l[0].ravel(),4:]
    XFA = X[l[1].ravel(),4:]
    pHit = np.row_stack([pHit,np.sum(XHit,0)])
    pFA = np.row_stack([pFA,np.sum(XFA,0)])
            
        
        
        
        
        
        
        
        
        
        
        
        
