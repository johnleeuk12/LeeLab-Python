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
    
    L = np.zeros((N_trial,t_period+prestim))
    Rt ={}
    Rt = np.zeros((N_trial,1)) # reward time relative to stim onset, in seconds

    for tr in range(N_trial):
        stim_onset = int(np.round(D_ppc[n,3][0,D_ppc[n,2][tr,0]]*1e3))
        lick_onset = int(np.round(D_ppc[n,3][0,D_ppc[n,2][tr,3]]*1e3))
        lick_onset = lick_onset-stim_onset
        L[tr,:] = L_all[0,stim_onset-prestim-1:stim_onset+t_period-1]
        if  D_ppc[n,6][tr,0] >0:
            Rt[tr,0] = int(D_ppc[n,6][tr,0]*1e3-stim_onset)
        
        # reformatting lick rates
    L2 = []
    for w in range(int((t_period+prestim)/window)):
        l = np.sum(L[:,range(window*w,window*(w+1))],1)
        L2 = np.concatenate((L2,l)) 
            
    L2 = np.reshape(L2,(int((t_period+prestim)/window),N_trial)).T
    # fig, axes = plt.subplots(1,1,figsize = (10,10))
    # axes.plot(np.mean(L3,0))


    X = D_ppc[n,2][:,2:6] # task variables
    
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
            
                       

                
    # select analysis and model parameters with c_ind
    
    if c_ind == 0:             
    # remove conditioning trials 
        Y = np.concatenate((Y[0:200,:],Y[D_ppc[n,4][0][0]:,:]),0)
        X = np.concatenate((X[0:200,:],X[D_ppc[n,4][0][0]:,:]),0)
        L2 = np.concatenate((L2[0:200,:],L2[D_ppc[n,4][0][0]:,:]),0)
    elif c_ind == 3:
    # only contain conditioning trials    

        c1 = 200
        c2 = D_ppc[n,4][0][0] + 25
        Y = Y[c1:c2]
        X = X[c1:c2]
        L2 = L2[c1:c2]
        Rt = Rt[c1:c2]
    elif c_ind == 1:
        c1 = 0
        c2 = 200
        Y = Y[c1:c2]
        X = X[c1:c2]
        L2 = L2[c1:c2]
        Rt = Rt[c1:c2]

    elif c_ind == 2:
        Y = Y[D_ppc[n,4][0][0]:,:]
        X = X[D_ppc[n,4][0][0]:,:]
        L2 = L2[D_ppc[n,4][0][0]:,:]
        Rt = Rt[D_ppc[n,4][0][0]:,:]

        

    
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
    
    X2 = np.column_stack([Xprestim,X[:,0],
                          X[:,2]*X[:,1],Xpre]) 


    L3 = (L2>0)

    
    return X2,Y, L3, Rt

# %% glm_per_neuron function code

def glm_per_neuron(n,t_period2,prestim,window,k,c_ind,ca,fig_on): 
    # if using spike data
    if ca == 0:
        X, Y, Y2,L = import_data_w_spikes(n,prestim,t_period2,window,c_ind)
    else:
    # if using Ca data
        X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period2,window,c_ind)
        Y2 = Y
    
    
    # t_period = (t_period+prestim)*2
    Nvar = 4
    t_period = t_period2 + prestim
    Yhat = [];
    Yhat_single = {};
    for it in np.arange(Nvar+1):
        Yhat_single[it] = []
        
    TT2 = [];
    Intercept = [];
    CI2 = [];
    score = [];
    N_trial2 = np.size(X,0)
    ED = 10
    ED2 = 5
    kernel_st = np.zeros((1,np.size(Y,1)))
    kernel_st[0,int(prestim/window):int((prestim+500)/window)] =1 # stim duration is 0.5s
    kernel_re = np.zeros((N_trial2,np.size(Y,1))) 
    for tr in np.arange(N_trial2):
        if Rt[tr,0] >0:
            kernel_re[tr,int((prestim+Rt[tr,0])/window):int((prestim+Rt[tr,0]+2000)/window)] = 1 #reward consumption is set to 2s
    # reg = TweedieRegressor(power = 0, alpha = 0)
    reg = ElasticNet(alpha = 4*1e-2, l1_ratio = 0.5) #Using a linear regression model with Ridge regression regulator set with alpha = 1
    # reg = Ridge(alpha = 4*1e-2)
    for w in range(int(t_period/window)):
        Xv = {}
        Xv1 = {}
        #Xv 1 : rh, 2: st 3: re, 4: lick
        for it in np.arange(3):
            Xv[it] = np.zeros((N_trial2,ED))
            Xv1[it] = np.zeros((N_trial2,ED))
        
        Xv[3] = np.zeros((N_trial2,ED+ED2))
        Xv1[3] = np.zeros((N_trial2,ED+ED2))
        y = Y2[:,w]
        # adding lag to lick
        if w < ED:
            for lag in np.arange(w):
                # l[:,lag] = L[:,w-lag]
                Xv[3][:,lag] = L[:,w-lag]
            for lag in np.arange(w,ED):
                # l[:,lag] = L[:,w]
                Xv[3][:,lag] = L[:,w]*0
        else:
            for lag in np.arange(ED):
                # l[:,lag] = L[:,w-lag]
                Xv[3][:,lag] = L[:,w-lag]
        # history variables         
        # X_sh = (np.ones_like(X_sh).T*[X[:,0]]).T
        Xv[0] = (np.ones_like(Xv[0]).T*[X[:,3]]).T
        
        if w+ED2<int(t_period/window):
            for lag in np.arange(ED2):
                Xv[3][:,ED+lag] = L[:,w+lag]
        else:
            for lag in np.arange(int(t_period/window)-w):
                Xv[3][:,ED+lag] = L[:,w+lag]
                
        
        if w*window > prestim: # if timeframe after stim onset
            for lag in np.arange(ED):
                Xv[1][:,lag] = kernel_st[0,w-lag]*X[:,1]
                Xv[2][:,lag] = kernel_re[:,w-lag]*X[:,2]
                
        # l = np.zeros((N_trial2,ED))



        # === initial run ===
        compare_score = {}
        for it in np.arange(5): # currently we use 4 variables, Rh, St, R, Lick
            if it ==4:
                 for itt in np.arange(4):
                     if itt == 3:
                         Xv1[itt] = np.zeros((N_trial2,ED+ED2))  
                     else:
                        Xv1[itt] = np.zeros((N_trial2,ED))  
            else:     
                Xv1[it] = Xv[it]                
                for itt in np.delete(np.arange(4),it):
                     if itt == 3:
                         Xv1[itt] = np.zeros((N_trial2,ED+ED2))  
                     else:
                        Xv1[itt] = np.zeros((N_trial2,ED))  
                        
            X2 = np.column_stack([np.ones_like(y),Xv1[0],Xv1[1],Xv1[2],Xv1[3]])
            X4 = np.column_stack([Xv1[0],Xv1[1],Xv1[2],Xv1[3]])            
            ss= ShuffleSplit(n_splits=k, test_size=0.30, random_state=0)
            cv_results = cross_validate(reg, X4, y, cv = ss , 
                                        return_estimator = True, 
                                        scoring = 'r2')  
            compare_score[it] = cv_results['test_score']
            compare_score[it] = compare_score[it]*(compare_score[it]>0)
            
            theta = np.zeros((np.size(X2,1)-1,k))
            inter = np.zeros((1,k))
            pp = 0
            for model in cv_results['estimator']:
                theta[:,pp] = model.coef_ 
                inter[:,pp] = model.intercept_
                pp = pp+1
            theta3 = np.concatenate((np.mean(inter,1),np.mean(theta,1)))
            yhat = X2 @theta3
            Yhat_single[it] = np.concatenate((Yhat_single[it],yhat))
            
            
        f = np.zeros((1,4))
        p = np.zeros((1,4))
        for it in np.arange(4):
            f[0,it], p[0,it] = stats.ks_2samp(compare_score[it],compare_score[4],alternative = 'less')
        max_it = np.argmax(f)
        init_score = compare_score[max_it]
        
     
        if p[0,max_it] > 0.05:
            max_it = []
        else:  
            # === stepwise forward regression ===
            step = 0
            while step < Nvar:
                max_ind = {}
                compare_score = {}
                f = np.zeros((1,Nvar))
                p = np.zeros((1,Nvar))
                for it in np.arange(Nvar):
                    m_ind = np.unique(np.append(max_it,it))
                    
                    for itt in m_ind:
                        Xv1[itt] = Xv[itt]
                    
                    for itt in np.delete(np.arange(Nvar),m_ind):
                        if itt == 3:
                            Xv1[itt] = np.zeros((N_trial2,ED+ED2))  
                        else:
                           Xv1[itt] = np.zeros((N_trial2,ED))  
                    
                    X4 = np.column_stack([Xv1[0],Xv1[1],Xv1[2],Xv1[3]])            
                    ss= ShuffleSplit(n_splits=k, test_size=0.30, random_state=0)
                    cv_results = cross_validate(reg, X4, y, cv = ss , 
                                                return_estimator = True, 
                                                scoring = 'r2')  
                    compare_score[it] = cv_results['test_score']
                    compare_score[it] = compare_score[it]*(compare_score[it]>0)
    
                    f[0,it], p[0,it] = stats.ks_2samp(compare_score[it],init_score,alternative = 'less')
                    
                max_ind = np.argmax(f)
                if p[0,max_ind] > 0.05 or p[0,max_ind] == 0:
                    step = Nvar
                else:
                    max_it = np.unique(np.append(max_it,max_ind))
                    init_score = compare_score[max_ind]
                    step += 1
     
        # === running regression with max_it ===
        if np.size(max_it) >1:
            for it in max_it:
                Xv1[it] = Xv[it]
            for it in np.delete(np.arange(Nvar),max_it):
                     if it == 3:
                         Xv1[it] = np.zeros((N_trial2,ED+ED2))  
                     else:
                        Xv1[it] = np.zeros((N_trial2,ED))  
        elif np.size(max_it) == 1:
            Xv1[max_it] = Xv[max_it]
            for it in np.delete(np.arange(Nvar),max_it):
                     if it == 3:
                         Xv1[it] = np.zeros((N_trial2,ED+ED2))  
                     else:
                        Xv1[it] = np.zeros((N_trial2,ED))       
        else:
            for it in np.arange(Nvar):
                     if it == 3:
                         Xv1[it] = np.zeros((N_trial2,ED+ED2))  
                     else:
                        Xv1[it] = np.zeros((N_trial2,ED))    
                
        X2 = np.column_stack([np.ones_like(y),Xv1[0],Xv1[1],Xv1[2],Xv1[3]])
        X4 = np.column_stack([Xv1[0],Xv1[1],Xv1[2],Xv1[3]])            
        ss= ShuffleSplit(n_splits=k, test_size=0.30, random_state=0)
        cv_results = cross_validate(reg, X4, y, cv = ss , 
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
        
        score = np.concatenate((score, cv_results['test_score']))
        TT2 = np.concatenate((TT2,np.mean(theta,1)))
        Intercept = np.concatenate((Intercept,np.mean(inter,1)))
        CI2 = np.concatenate((CI2,stats.sem(theta,1)))

        Yhat = np.concatenate((Yhat,yhat))
        
        
    Yhat = np.reshape(Yhat,(int(t_period/window),N_trial2)).T
    for it in np.arange(Nvar+1):
        Yhat_single[it] = np.reshape(Yhat_single[it],(int(t_period/window),N_trial2)).T
    
    TT2 = np.reshape(TT2,(int(t_period/window),np.size(X4,1))).T
    CI2 = np.reshape(CI2,(int(t_period/window),np.size(X4,1))).T
    score = np.reshape(score,(int(t_period/window),k))
    # 
    
    
    # Figures
    if fig_on ==1:
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))
        
            
        if c_ind == 3 or c_ind == 1 or c_ind == 2 :
            # cmap = ['tab:cyan','tab:pink','tab:blue','tab:red','orange']
            # clabels = ["Shistory","Rhistory","stim","reward","lick",]
            # lstyles = ['solid','solid','solid','solid','dotted']
            cmap = ['tab:pink','tab:blue','tab:red','orange']
            clabels = ["Rhistory","stim","reward","lick",]
            lstyles = ['solid','solid','solid','solid','dotted']
            
            
            
        x_axis = np.arange(1,t_period,window)
        
        X3 = np.row_stack([np.mean(TT2[0:ED-1,:],0)     #Shist
                            ,np.mean(TT2[ED:2*ED-1,:],0)  # Rhist
                            ,np.mean(TT2[2*ED:3*ED-1,:],0)  # stim
                            ,np.mean(TT2[3*ED:,:],0)])  # reward
                            # ,np.mean(TT2[4*ED:,:],0)])#lick
        CI3 = np.row_stack([np.mean(CI2[0:ED-1,:],0)     #Shist
                            ,np.mean(CI2[ED:2*ED-1,:],0)  # Rhist
                            ,np.mean(CI2[2*ED:3*ED-1,:],0)  # stim
                            # ,np.mean(CI2[3*ED:4*ED-1,:],0)  # reward
                            ,np.mean(CI2[3*ED:,:],0)])#lick
        
        for c in range(np.size(X3,0)):        
            ax2.plot(x_axis,ndimage.gaussian_filter(X3[c,:],1),linewidth = 2.0,
                     color = cmap[c], label = clabels[c], linestyle = lstyles[c])
            ax2.fill_between(x_axis,(ndimage.gaussian_filter(X3[c,:],1) - CI3[c,:]),
                            (ndimage.gaussian_filter(X3[c,:],1)+ CI3[c,:]), color=cmap[c], alpha = 0.2)
        
        # ax2.legend(loc = 'upper right')
    
        # e_lines = np.array([0,500,500+int(D_ppc[n,3]),2500+int(D_ppc[n,3])])
        e_lines = np.array([0,500,500+1000,2500+1000])
        # e_lines = np.array([0,500,500+1000,2500+1000, 8000, 8000+500, 8000+1500, 8000+ 3500])

        e_lines = e_lines+prestim
    
        
        ax2.vlines(x =e_lines, 
                  ymin = np.amin(ndimage.gaussian_filter(X3,sigma = [0,3])), 
                  ymax = np.amax(ndimage.gaussian_filter(X3,sigma = [0,3])),
                  linestyles = 'dashed',
                  colors = 'black', 
                  linewidth = 2.0)
        
        ax4.plot(x_axis,ndimage.gaussian_filter(np.median(score,1)*1e2,1))
        
        var_top = min(max(ndimage.gaussian_filter(np.median(score,1)*1e2,1)),100)
            
        # Plotting firing rates for one condition VS the other
        # 0 : contingency 
        # 1 : lick vs no lick
        # 2 : correct vs wrong
        # 3 : stim 1 vs stim 2
        # if c_ind ==0:
        #    # stim_ind = X3[:,3] == 1 
        # else:
        stim_ind1 = X[:,1] == 1   
        stim_ind2 = X[:,1] == 0 
        
        y1 = ndimage.gaussian_filter(np.mean(Y[stim_ind1,:],0),0)
        y2 = ndimage.gaussian_filter(np.mean(Y[stim_ind2,:],0),0)
        s1 = np.std(Y[stim_ind1,:],0)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(Y[stim_ind2,:],0)/np.sqrt(np.sum(stim_ind2))
        
        ax1.plot(x_axis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = lstyles[3])
        ax1.fill_between(x_axis,y1-s1, y1+s1, color = "blue",alpha = 0.5)
        ax1.plot(x_axis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = lstyles[3])
        ax1.fill_between(x_axis,y2-s2, y2+s2, color = "red",alpha = 0.5)
        # it= 3
        ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Yhat[stim_ind1,:],0),2),
                   linewidth = 2.0, color = "blue",linestyle = lstyles[4])
        ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Yhat[stim_ind2,:],0),2),
                   linewidth = 2.0, color = "red",linestyle = lstyles[4]) 
        # ax3.set_title('Prediction y_hat')
        
        
        ax1.set_title('Firing rate y')
        ax1.legend(loc = 'upper right')
    
        ax3.plot(x_axis,np.mean(L[stim_ind1,:],0),linewidth = 2.0, color = "blue",label = '10kHz',linestyle = lstyles[3])
        ax3.plot(x_axis,np.mean(L[stim_ind2,:],0),linewidth = 2.0, color = "red",label = '5kHz',linestyle = lstyles[3])

        
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
    
    return X, Y, Yhat, Model_Theta, score, Intercept,Yhat_single

# %% Main
        





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
prestim = 3000

window = 100 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 20 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [1]



if ca ==0:
    D_ppc = load_matfile()
    # good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    good_list = find_good_data_Ca(t_period)
    
# %% Run GLM 
Data = {}
# Data = np.load('R1lick_1221.npy', Data,allow_pickle= True).item()
# 
# additional code for explained variance comparison
DataS = {}
S = np.zeros((1,5))
ana_period = np.array([0, t_period+prestim])
weight_thresh = 2*1e-2

# change c_ind and n here. 

for c_ind in c_list:
    # t = 0 
    good_list2 = [];
    for n in np.arange(np.size(D_ppc,0)):
        
        n = int(n)
        if D_ppc[n,4][0][0] > 0:
            # X, Y, Yhat, Model_Theta, score = glm_per_neuron(n, t_period, prestim, window,k,c_ind)
            # Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score} 
            try:
                # maxS = build_model(n, t_period, prestim, window, k, c_ind, ca)
                # maxS = Data[n,c_ind-1]["maxS"]
                # maxS = [0,1,2,3]   
                X, Y, Yhat, Model_Theta, score, intercept,Yhat_single = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,1)
                Data[n,c_ind-1] = {"X":X,"coef" : Model_Theta, "score" : score, 'Y' : Y,'Yhat' : Yhat, "intercept" : intercept,'Yhat_single' : Yhat_single }
                # t += 1
                # print(t,"/",len(good_list))
                good_list2 = np.concatenate((good_list2,[n]))
                print(n)
                
            except KeyboardInterrupt:
                
                break
            except:
            
                print("Error, probably not enough trials") 
# np.save('R2lick_1221.npy', Data,allow_pickle= True)      
#
# %% for each weight, get corresponding yhat
# Calculating R2 per neuron
# good_list = good_list_int
good_list = np.arange(np.size(D_ppc,0))
d_list = good_list > 179
# d_list = good_list > 118
d_list3 = good_list <= 179
# d_list3 = good_list <= 118

# c_ind = 2
ax_sz = 3

good_list_sep = good_list[d_list3]

# Rscore = {}
Rscore = np.zeros((ax_sz+2,np.size(good_list)))
    
y_lens = np.arange(int((t_period+prestim)/window))


   
for n in np.arange(np.size(good_list,0)):
        # print(n)
    nn = good_list[n]
    nn = int(nn)
    # maxS = Data[nn,c_ind-1]["maxS"]
    if D_ppc[n,4][0][0] > 0:
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
        
        XX, YY, L2, Rt = import_data_w_Ca(D_ppc,nn,prestim,t_period,window,c_ind)
        
        X3 = np.row_stack([np.sum(Model_Theta[0:9,:],0)     #Shist
                               ,np.sum(Model_Theta[10:19,:],0)  # Rhist
                               ,np.sum(Model_Theta[20:29,:],0)  # stim
                               #,np.sum(Model_Theta[30:39,:],0)  # reward
                               ,np.sum(Model_Theta[30:,:],0)])#lick
        # CI3 = np.row_stack([np.mean(CI2[0:9,:],0)     #Shist
        #                        ,np.mean(CI2[10:19,:],0)  # Rhist
        #                        ,np.mean(CI2[20:29,:],0)  # stim
        #                        ,np.mean(CI2[30:39,:],0)  # reward
        #                        ,np.mean(CI2[40:49,:],0)])#lick
    
        
        theta3 = np.concatenate(([ymean[0,:]],X3[:-1,y_lens]),0)
        X2 = np.column_stack([np.ones((np.size(X,0),1)),X[:,3],X[:,1],X[:,2]])
        # for f in np.arange(ax_sz):
        #     yhat2 = X2[:,[0,f+1]] @ theta3[[0,f+1],:]
        #     # yhat2 = X2[:,:] @ theta3[:,:] + L2*X3[ax_sz,:]
            
        #     # if f == 1:
        #     #     fig, axes  = plt.subplots(1,1, figsize = (10,10))
        #     #     axes.plot(x_axis,np.mean(Yhat[(X2[:,f+1] ==0),:],0), color = "red",linestyle = "dotted")
        #     #     axes.plot(x_axis,np.mean(Yhat[(X2[:,f+1] ==1),:],0), color = "blue",linestyle = "dotted")
        #     #     axes.plot(x_axis,np.mean(Y[(X2[:,f+1] ==0),:],0), color = "red",linestyle = "solid")
        #     #     axes.plot(x_axis,np.mean(Y[(X2[:,f+1] ==1),:],0), color = "blue",linestyle = "solid")
        #     #     axes.plot(x_axis,np.mean(ymean,0))
                
        #     Rscore[f,n] = 1- np.sum(np.square(Y-yhat2))/np.sum(np.square(Y-ymean))
        #     if Rscore[f,n] ==0:
        #         Rscore[f,n] = -1
        #     # fig, axes = plt.subplots(1,1, figsize = (10,10))    
        #     # axes.plot(np.mean(yhat2[(X2[:,3] == 1),:],0), color = "blue" )
        #     # axes.plot(np.mean(yhat2[(X2[:,3] == 0),:],0), color = "red" )
        
    
        # yhat2 = X2[:,[0]] @ theta3[[0],:] + L2*X3[ax_sz,:]
        # Rscore[ax_sz,n] = 1- np.sum(np.square(Y-yhat2))/np.sum(np.square(Y-ymean))
        # if Rscore[ax_sz,n] ==0:
        #     Rscore[ax_sz,n] = -1
                    
                    
        Rscore[ax_sz+1,n] = 1- np.sum(np.square(Y-Yhat))/np.sum(np.square(Y-ymean))
            # Rscore[c_ind][:,n]    

# scatter_ind = [np.arange(ax_sz+1)]*np.ones((ax_sz+1,len(good_list))).T
# scatter_ind = scatter_ind.T


# %% Rscore method, 2
good_list = np.arange(np.size(D_ppc,0))
d_list = good_list > 179
# d_list = good_list > 118
d_list3 = good_list <= 179
# d_list3 = good_list <= 118

# c_ind = 2
ax_sz = 4

good_list_sep = good_list[d_list3]

# Rscore = {}
Rscore = np.zeros((ax_sz+1,np.size(good_list)))
    
y_lens = np.arange(int((t_period+prestim)/window))


   
for n in np.arange(np.size(good_list,0)):
        # print(n)
    nn = good_list[n]
    nn = int(nn)
    # maxS = Data[nn,c_ind-1]["maxS"]
    if D_ppc[n,4][0][0] > 0:
        Y = Data[nn,c_ind-1]["Y"][:,:]
        Yhat = Data[nn,c_ind-1]["Yhat"][:,:]
        Model_Theta = Data[nn,c_ind-1]["coef"]
        intercept = Data[nn,c_ind-1]["intercept"]
        ymean = np.ones((len(y_lens),np.size(X,0))).T*intercept        
        for f in np.arange(ax_sz):         
            yhat2 = Data[nn,c_ind-1]["Yhat_single"][f]               
            Rscore[f,n] = 1- np.sum(np.square(Y-yhat2))/np.sum(np.square(Y-ymean))
            if Rscore[f,n] ==0:
                Rscore[f,n] = -1
                                      
        Rscore[ax_sz,n] = 1- np.sum(np.square(Y-Yhat))/np.sum(np.square(Y-ymean))


# %% plot R score


cmap = ['tab:pink','tab:blue','tab:red','orange']
# c_ind = -1

# d_list = good_list > 118
d_list3 = good_list <= 179
# d_list3 = good_list <= 118

d_list = good_list > 179

ax_sz = 5
def make_RS(d_list):
    fig, axes = plt.subplots(1,1, figsize = (10,8))
    Rsstat = {}
    for f in np.arange(0,np.size(Rscore,0)-1):
        Rs = Rscore[f,d_list]
        # Rmax = Rscore[np.size(Rscore,0)-1,d_list]
        # Rmax = Rmax[Rs>0.02]
        Rs = Rs[Rs>0.02]
    
        # Rs = Rs/(Rmax+0.03)
        Rsstat[c_ind,f] = Rs
        axes.scatter(np.ones_like(Rs)*(f+(c_ind+1)*-0.3),Rs,c = cmap[f])
        axes.scatter([(f+(c_ind+1)*-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')
            # axes.boxplot(Rs,positions= [f+(c_ind+1)*-0.3])
    axes.scatter(np.ones_like(Rscore[ax_sz-1,d_list])*(ax_sz-1+(c_ind+1)*-0.3),Rscore[ax_sz-1,d_list])
    axes.scatter([(ax_sz-1+(c_ind+1)*-0.3)],np.mean(Rscore[ax_sz-1,d_list]),c = 'k',s = 500, marker='_')
        
    Rsstat[c_ind,4] = Rscore[4,d_list]
    
        # axes.boxplot(Rscore[c_ind][4,d_list3],positions= [4+(c_ind+1)*-0.3])
    # axes.set_ylim([-0.05,0.3])
    # plt.savefig("PPC_AC.svg")

    return Rsstat


RsStat_PIC = make_RS(d_list3)
RsStat_PAC = make_RS(d_list)



bins = np.arange(0,0.5,0.025)
fig, axes = plt.subplots(1,1,figsize = (5,5))
axes.hist(RsStat_PAC[c_ind,4], bins , alpha=0.7, rwidth=1)
axes.hist(RsStat_PIC[c_ind,4], bins , alpha=0.7, rwidth=1)
# axes.set_xlim([0.005,0.25])
axes.set_ylim([0,100])
np.mean((RsStat_PAC[c_ind,4][RsStat_PAC[c_ind,4]>0.02]))
# 

# stats.ks_2samp(RsStat_PAC[-1,4],RsStat_PAC2[-1,4])
# plt.savefig("Rscore_hist.svg")
# 


good_listR = Rscore[4,:] > 0.02
good_listR[22] = False
good_listR[51] = False
# good_listR[67] = False
# good_listR[45] = False
good_listRu = good_list[good_listR]

# good_listRu[43]

# %% plot example neurons 



x_axis = np.arange(1, prestim+t_period, window)
x_axis = (x_axis-prestim)*1e-3

def plt_ex_neurons(nn,c1,c2):    
    # X, Y, Yhat, Model_Theta, score, intercept = glm_per_neuron(nn, t_period, prestim, window,k,c_ind,ca,1)
        # Data[nn,c_ind-1] = {"X" : X,"coef" : Model_Theta, "intercept" : intercept, "score" : score, 'Y' : Y,'Yhat' : Yhat, 'maxS': maxS}
    y_lens = np.arange(int((t_period+prestim)/window))    
    Y = Data[nn,c_ind-1]["Y"][:,:]
    X = Data[nn,c_ind-1]["X"][:,:]
    Yhat = Data[nn,c_ind-1]["Yhat"][:,:]
    Model_Theta = Data[nn,c_ind-1]["coef"]
    intercept = Data[nn,c_ind-1]["intercept"]
    ymean = np.ones((len(y_lens),np.size(X,0))).T*intercept
    
    X4 = D_ppc[nn,2][:,2:6] # task variables
    X4 = X4[c1:c2,:]
    
    ### divide into Hit, Miss, FA and CR
    X5 = np.column_stack([(X4[:,0] == 1) * (X4[:,1] == 1), # Hit
                           (X4[:,0] == 1) * (X4[:,1] == 0), # MIss
                           (X4[:,0] == 0) * (X4[:,1] == 1), # FA
                           (X4[:,0] == 0) * (X4[:,1] == 0)]) # CR
    X5 = np.column_stack([X5,np.concatenate(([False],X5[:-1,0]),0),np.concatenate(([False],X5[:-1,1]),0)])
    ### initialize kernels ###
    
    ED = 10
    N_trial2 = c2-c1
    XX, YY, L, Rt = import_data_w_Ca(D_ppc,nn,prestim,t_period,window,c_ind)
    kernel_st = np.zeros((1,np.size(Y,1)))
    kernel_st[0,int(prestim/window):int((prestim+500)/window)] =1 # stim duration is 0.5s
    kernel_re = np.zeros((N_trial2,np.size(Y,1))) 
    for tr in np.arange(N_trial2):
        if Rt[tr,0] >0:
            kernel_re[tr,int((prestim+Rt[tr,0])/window):int((prestim+Rt[tr,0]+2000)/window)] = 1 #reward consumption is set to 2s
    
    
    ## Model weights W3 ##
    yh = {}
    for it in np.arange(4):
        yh[it] = np.zeros((N_trial2,int((t_period+prestim)/window)))
    for w in range(int((t_period+prestim)/window)):
        Xv = {}
        # Xv1 = {}
        #Xv 1 : rh, 2: st 3: re, 4: lick
        for it in np.arange(4):
            Xv[it] = np.zeros((N_trial2,ED))
            # Xv1[it] = np.zeros((N_trial2,ED))
        
        # adding lag to lick
        if w < ED:
            for lag in np.arange(w):
                # l[:,lag] = L[:,w-lag]
                Xv[3][:,lag] = L[:,w-lag]
            for lag in np.arange(w,ED):
                # l[:,lag] = L[:,w]
                Xv[3][:,lag] = L[:,w]
        else:
            for lag in np.arange(ED):
                # l[:,lag] = L[:,w-lag]
                Xv[3][:,lag] = L[:,w-lag]
        # history variables         
        # X_sh = (np.ones_like(X_sh).T*[X[:,0]]).T
        Xv[0] = (np.ones_like(Xv[0]).T*[X[:,3]]).T
        
        if w*window > prestim: # if timeframe after stim onset
            for lag in np.arange(ED):
                Xv[1][:,lag] = kernel_st[0,w-lag]*X[:,1]
                Xv[2][:,lag] = kernel_re[:,w-lag]*X[:,2]
        
        for it in np.arange(4): 
            yh[it][:,w] = Xv[it] @ Model_Theta[it*ED:it*ED+ED,w]
        
    
    
    fig, axes = plt.subplots(6,6,figsize = (30,20))
    for ind1 in np.arange(6):
        axes[0,ind1].plot(x_axis,np.mean(Y[X5[:,ind1],:],0),color = "blue",linewidth = 3.0)
        axes[0,ind1].plot(x_axis,np.mean(Yhat[X5[:,ind1],:],0),color = "red",linewidth = 3.0)
        axes[1,ind1].plot(x_axis,np.mean(ymean[X5[:,ind1],:],0),color = "black",linewidth = 3.0)
        axes[0,ind1].xaxis.set_tick_params(labelsize=20)
        axes[0,ind1].yaxis.set_tick_params(labelsize=20)
        axes[1,ind1].xaxis.set_tick_params(labelsize=20)
        axes[1,ind1].yaxis.set_tick_params(labelsize=20)
        axes[0,ind1].set_ylim([np.min(Yhat)*1.2,np.max(Yhat)*1.2])
    
    pltmin = np.min([np.mean(yh[0],0),np.mean(yh[1],0),np.mean(yh[2],0),np.mean(yh[3],0)])
    pltmax = np.max([np.mean(yh[0],0),np.mean(yh[1],0),np.mean(yh[2],0),np.mean(yh[3],0)])
    for ind2 in np.arange(2,6):
        for ind1 in np.arange(6):
            # yhat_tv = 
            # axes[ind2,ind1].plot(x_axis,np.mean(yh[ind2-2][X5[:,ind1],:],0)+np.mean(ymean,0),color = cmap[ind2-2])
            axes[ind2,ind1].plot(x_axis,np.mean(yh[ind2-2][X5[:,ind1],:],0),color = cmap[ind2-2],linewidth = 3.0)
            
            axes[ind2,ind1].set_ylim([pltmin*3,pltmax*3])
            axes[ind2,ind1].yaxis.set_tick_params(labelsize=20)
            axes[ind2,ind1].xaxis.set_tick_params(labelsize=20)
    return yh        



nn = 474
c1 = 0
c2 = 200                              
yh = plt_ex_neurons(nn,c1,c2)    
    
    # %% Normalized population average of task variable weights
# c_ind = 1
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

# cmap = ['tab:cyan','tab:pink','tab:blue','tab:red','orange']
# clabels = ["Shistory","Rhistory","stim","reward","lick"]
# lstyles = ['solid','solid','solid','solid','dotted']

cmap = ['tab:pink','tab:blue','tab:red','orange']
clabels = ["Rhistory","stim","reward","lick"]
lstyles = ['solid','solid','solid','dotted']
ax_sz = len(cmap)

score = np.zeros((70,1)) 
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
    X3 = np.row_stack([np.sum(Model_coef[0:9,:],0)     #Shist
                               ,np.sum(Model_coef[10:19,:],0)  # Rhist
                               ,np.sum(Model_coef[20:29,:],0)  # stim
                               # ,np.sum(Model_coef[30:39,:],0)  # reward
                               ,np.sum(Model_coef[30:,:],0)])#lick
    # Model_coef = np.abs(Model_coef)/(np.max(np.abs(Model_coef)) + 0.1) # soft normalization value for model_coef
    Model_coef = X3/(np.max(np.abs(X3)) + 0.2) # soft normalization value for model_coef
# 
    norm_score = np.median(Model_score, 1)
    norm_score[norm_score < weight_thresh] = 0
    # norm_score = ndimage.gaussian_filter(norm_score,2)
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
        axes.plot(x_axis*1e-3-prestim*1e-3,y,c = cmap[f],linestyle = lstyles[f])
        axes.fill_between(x_axis*1e-3-prestim*1e-3,y-error,y+error,facecolor = cmap[f],alpha = 0.3)
        axes.set_ylim([-0.01,0.08])

# axes[1].plot(x_axis*1e-3-prestim*1e-3,ndimage.gaussian_filter(np.mean(norm_score_all,0),2))

e_lines = np.array([0, 500, 500+1000, 2500+1000])
e_lines = e_lines+500


# 
# np.save('Conv_R2_350.npy', Convdata,allow_pickle= True)

# %% plotting weights by peak order
listOv = {}

f = 0
W5 = {}
W5AC= {}
W5IC = {}
max_peak3 ={}
tv_number = {}
for ind in [0,1]:
    for f in  [3]: #np.arange(4):
        list0 = (np.mean(Convdata[f],1) != 0)
        
        Lg = len(good_listRu)
        Lic = np.where(good_listRu <180)
        Lic = Lic[0][-1]
        if ind == 0:
            list0[Lic:Lg] = False # PPCIC
        elif ind == 1:           
            list0[0:Lic] = False # PPCAC
        
        # list0ind = np.arange(Lg)
        # list0ind = list0ind[list0]
        list0ind = good_listRu[list0]
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
            if SD< 0.05:
                SD = 0.05
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
        
        W3 = np.concatenate((W1,W2), axis = 0)
        tv_number[ind,f] = [np.size(W1,0),np.size(W2,0)]
        fig, axes = plt.subplots(1,1,figsize = (10,10))
        
        
        clim = [-0.0, 0.8]
        im1 = axes.imshow(W4[:,:], clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
        # im2 = axes[1].imshow(W2, aspect = "auto", interpolation = "None")
        
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im1, cax=cbar_ax)
        if ind == 0:
            W5IC[f] = W4
        elif ind == 1:           
            W5AC[f] = W4
        # W4IC = W4
    
# print(np.size(np.intersect1d(listOv[0],listOv[3])))
np.save('PPC_Hist.npy',listOv,allow_pickle = True)


list0n = good_listRu[list0]


# %% 




# %%
f =3
fig, axes = plt.subplots(1,1,figsize = (10,10))

y1 = ndimage.gaussian_filter1d(np.mean(W5AC[f],0),2)
error1 = np.std(W5AC[f],0)/np.sqrt(np.size(W5AC[f],0))
error3 = np.std(W5AC[f])/np.sqrt(np.size(W5AC[f][:,:],0))
axes.plot(x_axis*1e-3-prestim*1e-3,y1,c = cmap[f],linestyle = lstyles[f])
axes.fill_between(x_axis*1e-3-prestim*1e-3,y1-error1,y1+error1,facecolor = cmap[f],alpha = 0.3)
axes.set_ylim([0,0.35])
# for t in y_lens:
#     s,p = stats.ttest_1samp(W5AC[f][:,t],error3,alternative = "greater")
#     if p < 0.05:
#         axes.scatter([(t*window-prestim)*1e-3],[0.30],marker = '*', color = 'black')

# fig, axes = plt.subplots(1,1,figsize = (10,10))
y2 = ndimage.gaussian_filter1d(np.mean(W5IC[f],0),2)
error2 = np.std(W5IC[f],0)/np.sqrt(np.size(W5IC[f],0)) 
error3 = np.std(W5IC[f])/np.sqrt(np.size(W5IC[f][:,:],0))

axes.plot(x_axis*1e-3-prestim*1e-3,y2,c = cmap[f],linestyle = "dotted")
axes.fill_between(x_axis*1e-3-prestim*1e-3,y2-error2,y2+error2,facecolor = cmap[f],alpha = 0.3)
axes.set_ylim([0,0.35])
# for t in y_lens:
#     s,p = stats.ttest_1samp(W5IC[f][:,t],error3,alternative = "greater")
#     if p < 0.05:
#         axes.scatter([(t*window-prestim)*1e-3],[0.30],marker = '*', color = 'black')

# %% plot histogram
# fig, axes = plt.subplots(figsize=(6,4), nrows=1)
# W5 = W5IC
# W5 = W5AC
# for f in [0,3]: #np.arange(4):
#     y1 = np.mean(W5[f],0)*np.size(W5[f],0)/423
#     s1 = np.std(W5[f],0)/np.sqrt(423)
    
#     # y2 = np.mean(W2,0)
#     # s2 = np.std(W2,0)/np.sqrt(np.size(W2,0))
    
#     axes.plot(x_axis*1e-3-prestim*1e-3,y1,c = cmap[f],linestyle = 'solid')
#     axes.fill_between(x_axis*1e-3-prestim*1e-3,y1-s1,y1+s1,facecolor = cmap[f],alpha = 0.3)
#     axes.set_ylim([-0.01,0.15])
#     axes.set_xlim([-3,4])
    # plt.savefig("PIC.svg")
 
f  =0
ind = 1
bins = np.linspace(0,80,15)
fig, axes = plt.subplots(figsize=(6,4), nrows=1)
axes = sns.histplot(max_peak3[ind,f], stat='probability',bins = bins)
# axes.hist(max_peak3,bins = bins, density = "True", stacked = "True")
axes.set_ylim([0,0.10])

