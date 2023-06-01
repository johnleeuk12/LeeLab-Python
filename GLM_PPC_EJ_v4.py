# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:31:34 2023

        
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


# from matplotlib.colors import BoundaryNorm

# from sklearn.metrics import get_scorer_names, d2_tweedie_score, make_scorer


# %% File name and directory

# change fname for filename

# fname = 'PPC_GLM_dataset_AllSession_FR_230209.mat'
fname = 'CaData_all_withlicktime.mat'
fdir = 'D:\Python\Data'
# fname = 'GLM_dataset_220824_new.mat'


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
        # X2 = np.column_stack([X[:,0],(X[:,0]-1)*-1,
        #                      X[:,3],(X[:,3]-1)*-1,
        #                      X[:,2]*X[:,1],Xpre])
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
    # D_ppc = load_matfile_Ca()
    
    
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
    Rt =  D_ppc[n,5] # reward time relative to stim onset, in seconds
    t_period = t_period+prestim
    
    # re-formatting Ca traces
    Yraw = {}
    Yraw = D_ppc[n,0]
    
    
    
    # Yraw2 = ndimage.median_filter(Yraw, size = 3000)
    # Y_median = sliding_median(Yraw[0,:],200)
    Yraw2 = np.concatenate((np.flip(Yraw[0,0:3000],0),Yraw[0,:],Yraw[0,-3000:-1]),0)
    sliding_w= np.lib.stride_tricks.sliding_window_view(np.arange(np.size(Yraw,1)+6000), 6000)
    Ymed_wind = np.zeros((1,np.size(Yraw,1)))
    for s in np.arange(np.size(Yraw,1)):
        Ymed_wind[0,s] = np.median(Yraw2[sliding_w[s,:]])
        
    Yraw3 = Yraw-Ymed_wind+np.mean(Yraw)
    
    # fig, axes = plt.subplots(1,1)
    # axes.plot(np.arange(85141),ndimage.gaussian_filter(Yraw[0,:],1000))
    # axes.plot(np.arange(85141),ndimage.gaussian_filter(Yraw3[0,:],1000))    
    
    # Yraw[0,100:-99] = (Yraw[0,100:-99]-Y_median)/Y_median
    
    Y = np.zeros((N_trial,int(t_period/window)))
    for tr in range(N_trial):
        Y[tr,:] = Yraw3[0,D_ppc[n,2][tr,0]-1 - int(prestim/window): D_ppc[n,2][tr,0] + int(t_period/window)-1 - int(prestim/window)]
    
    
    # for t in np.arange(int(t_period/window)):
    #     Y[:,t] = Y[:,t]- np/median(Y[:,t])


                
    # select analysis and model parameters with c_ind
    
    if c_ind != 3:             
    # remove conditioning trials 
        Y = np.concatenate((Y[0:200,:],Y[D_ppc[n,4][0][0]:,:]),0)
        X = np.concatenate((X[0:200,:],X[D_ppc[n,4][0][0]:,:]),0)
        L2 = np.concatenate((L2[0:200,:],L2[D_ppc[n,4][0][0]:,:]),0)
    else:
    # only contain conditioning trials    
        Y = Y[201:D_ppc[n,4][0][0]]
        X = X[201:D_ppc[n,4][0][0]]
        L2 = L2[201:D_ppc[n,4][0][0]]

    
    # Add reward  history
    Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
    Xpre = Xpre[:,None]
    Xpre2 = np.concatenate(([0,0],X[0:-2,2]*X[0:-2,1]),0)
    Xpre2 = Xpre2[:,None]
    # Add reward instead of action
    X2 = np.column_stack([X[:,0],X[:,3],
                          X[:,2]*X[:,1],Xpre]) 

    

    
    return X2,Y, L2, Rt
    
# %% Main function for GLM
# %% glm_per_neuron function code

def glm_per_neuron(n,t_period,prestim,window,k,c_ind,ca, m_ind,fig_on): 
    # if using spike data
    if ca == 0:
        X, Y, Y2,L = import_data_w_spikes(n,prestim,t_period,window,c_ind)
    else:
    # if using Ca data
        X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
        Y2 = Y
    
    
    t_period = t_period+prestim
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
        X3 = X
        Xm = np.zeros_like(X3)
        Xm[:,m_ind] = 1
        X3 = X3*Xm
        # adding kernels to each task variable
        if w*window <= prestim-window:
            X3[:,0:3] = 0;
        elif w*window <= prestim+1500-window:
            
            if ca == 0:
                X3[:,2]= 0;
            elif ca == 1:
                for tr in np.arange(np.size(L,0)):
                    if np.isnan(Rt[tr,0]):
                        X3[tr,2] = 0;
                    else:
                        if w*window <= prestim + Rt[tr,0]*1e3 -window:
                            X3[tr,2] = 0;
                        
        

        
        
        
        X2 = np.column_stack([np.ones_like(y),X3])
        ss= ShuffleSplit(n_splits=k, test_size=0.20, random_state=0)
        y2 = ndimage.gaussian_filter(y,0)
        cv_results = cross_validate(reg, X3, y2, cv = ss , 
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
        yhat1 = X2[0:200,:] @ theta3
        yhat2 = X2[200:,:] @ theta3
        
        
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
        elif c_ind == -2:
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
        stim_ind1 = X3[:,2] == 1     
        stim_ind2 = X3[:,2] == 0  
        
    
        ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind1,:],0),2),
                  linewidth = 2.0, color = cmap[2],label = 'Reward',linestyle = lstyles[3])
        ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind2,:],0),2),
                  linewidth = 2.0, color = cmap[2],label = 'No Reward',linestyle = lstyles[3])
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
        plt.show()
    Model_Theta = TT2
    
    return X3, Y, Yhat, Model_Theta, score, Intercept

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



t_period = 7000
prestim = 1000

window = 50 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 10 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [-2]



if ca ==0:
    D_ppc = load_matfile()
    good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    good_list = find_good_data_Ca(t_period)

# %%
Data = {}
Data = np.load('D:\Python\Data_PPCAll_Ca_05_31.npy',allow_pickle= True).item()

# additional code for explained variance comparison
DataS = {}
S = np.zeros((1,Nvar))
ana_period = np.array([0, t_period+prestim])
weight_thresh = 2*1e-2

# change c_ind and n here. 

for c_ind in c_list:
    # t = 0 
    good_list2 = [];
    for n in good_list:
        
        n = int(n)
        # X, Y, Yhat, Model_Theta, score = glm_per_neuron(n, t_period, prestim, window,k,c_ind)
        # Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score} 
        try:
            maxS = build_model(n, t_period, prestim, window, k, c_ind, ca)
            # maxS = Data[n,c_ind-1]["maxS"]
            # maxS = [0,1,2,3]   
            X, Y, Yhat, Model_Theta, score, intercept = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,maxS,1)
            Data[n,c_ind-1] = {"X":X,"coef" : Model_Theta, "score" : score, 'Y' : Y,'Yhat' : Yhat,'maxS' : maxS, "intercept" : intercept}
            # t += 1
            # print(t,"/",len(good_list))
            good_list2 = np.concatenate((good_list2,[n]))
            print(n)
            
        except KeyboardInterrupt:
            
            break
        except:
            
            print("Error, probably not enough trials") 
# np.save('Data_PPCall_Ca_05_31.npy', Data,allow_pickle= True)     

# %% for each weight, get corresponding yhat
# Calculating R2 per neuron
# good_list = good_list_int
d_list = good_list > 179

d_list3 = good_list <= 179

ax_sz = 4

good_list_sep = good_list[d_list3]

# Rscore = {}
Rscore = np.zeros((ax_sz+1,np.size(good_list)))
    
y_lens = np.arange(160)
   
for n in np.arange(np.size(good_list,0)):
        # print(n)
    nn = good_list[n]
    nn = int(nn)
    maxS = Data[nn,c_ind-1]["maxS"]
    try:
        X = Data[nn,c_ind-1]["X"]
        intercept = Data[nn,c_ind-1]["intercept"]

    except:                
        X, Y, Yhat, Model_Theta, score, intercept = glm_per_neuron(nn, t_period, prestim, window,k,c_ind,ca,maxS,1)
        Data[nn,c_ind-1] = {"X" : X,"coef" : Model_Theta, "intercept" : intercept, "score" : score, 'Y' : Y,'Yhat' : Yhat, 'maxS' : maxS}
        
    Y = Data[nn,c_ind-1]["Y"][:,y_lens]
    Yhat = Data[nn,c_ind-1]["Yhat"][:,y_lens]
    Model_Theta = Data[nn,c_ind-1]["coef"]
    ymean = np.ones((len(y_lens),np.size(X,0))).T*Data[nn,c_ind-1]["intercept"][y_lens]
        # ymean[0,:] = intercept
        
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

# %% plot R score


cmap = ['tab:purple','tab:blue','tab:red','tab:orange']
c_ind = -1

d_list = good_list > 179
# 
d_list3 = good_list <= 179
# d_list3 = good_list > 179


def make_RS(d_list):
    fig, axes = plt.subplots(1,1, figsize = (10,8))
    Rsstat = {}
    for f in np.arange(0,ax_sz):
        Rs = Rscore[f,d_list]
        Rmax = Rscore[4,d_list]
        Rmax = Rmax[Rs>0.01]
        Rs = Rs[Rs>0.01]
    
        # Rs = Rs/(Rmax+0.03)
        Rsstat[c_ind,f] = Rs
        axes.scatter(np.ones_like(Rs)*(f+(c_ind+1)*-0.3),Rs,c = cmap[f])
        axes.scatter([(f+(c_ind+1)*-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')
            # axes.boxplot(Rs,positions= [f+(c_ind+1)*-0.3])
    axes.scatter(np.ones_like(Rscore[4,d_list])*(4+(c_ind+1)*-0.3),Rscore[4,d_list])
    axes.scatter([(4+(c_ind+1)*-0.3)],np.mean(Rscore[4,d_list]),c = 'k',s = 500, marker='_')
        
    Rsstat[c_ind,4] = Rscore[4,d_list]
    
        # axes.boxplot(Rscore[c_ind][4,d_list3],positions= [4+(c_ind+1)*-0.3])
    axes.set_ylim([-0.05,0.2])

    return Rsstat


RsStat_PIC = make_RS(d_list3)
RsStat_PAC = make_RS(d_list)

good_listR = Rscore[4,:] > 0.02


# %% Calculating best_kernel


best_kernel = {}

"""
for each rule (c_ind) we have a best_kernel matrix
each matrix contains best time_bin (ind), best coefficient and normalized model_weights
row 1       :   best ind
row 2       :   best category [0 1 2 3 4 ] is ["Uncategorized", "Action", "Correct","Stimuli"]
row 3 to 5  :   normalized weights for action, correct and stimuli

when adding contingency
row 3 to 6  :   contingency, action, correct, stimuli

when adding trial history, the last row is trial history, with best category going up to 5 
"""

good_list2 = good_list


def get_best_kernel(b_ind, window, window2, Data, c_ind, ana_period,good_list):
    best_kernel[c_ind] = np.zeros((b_ind,np.size(good_list,0)))


    k = 0;
    for n in good_list2:
        n = int(n)
        mi, bs, coef,beta_weights,mean_score, var_score,score_pool = Model_analysis(n, window, window2, Data,c_ind,ana_period)
        norm_coef = np.abs(coef)
        # Y_mean = np.mean(Data[n,c_ind-1]["Y"])
        if bs > weight_thresh:
            best_kernel[c_ind][0,k] = int(mi)
            if np.max(np.abs(coef))>var_score[0,mi]:
                best_kernel[c_ind][1,k] = int(np.argmax(np.abs(coef)))+1
            for i in np.arange(np.size(coef)):
                best_kernel[c_ind][i+2,k] = norm_coef[i] 
                

            
        else:
            best_kernel[c_ind][2:b_ind,k] = np.ones((1,b_ind-2))*-1    
        k = k+1
        
    return best_kernel

weight_thresh = 2*1e-2


# Here we define the time period for model analysis. 
# ana_period = np.array([2000, 4000]) # (Stimulus presentation period)
# ana_period = np.array([1500, 2500])
# ana_period = np.array([2500, 4500])
ana_period = np.array([0, 6000])
for c_ind in c_list:
    if c_ind == 0 or c_ind ==-3 or c_ind == -4 or c_ind == -2:
        b_ind = 9
    elif c_ind == -1:
        b_ind = 7
    else:
        b_ind = 5
        
    best_kernel = get_best_kernel(b_ind, window, window2, Data, c_ind, ana_period,good_list)
    
    
# %% Normalized population average of task variable weights

d_list = good_list > 179

d_list3 = good_list <= 179

c_ind = -2
# good_list2 = good_list[d_list & good_listR]

# cat_list = best_kernel[c_ind][0,:] != 0 # Only neurons that were categorized

# good_list_sep = good_list[cat_list]
# good_list_sep = good_list[d_list & good_listR]
good_list_sep = good_list[:]


weight_thresh = 2*1e-2


if c_ind == 0 or c_ind == -2:
    cmap3 = ['tab:purple','tab:blue','tab:red','tab:olive']
    ax_sz = len(cmap3)
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
    norm_score = ndimage.gaussian_filter(norm_score,1)
    # norm_score[norm_score > 0] = 1 
    # if np.max(norm_score)>0:
    #     norm_score = norm_score/(np.max(norm_score)+weight_thresh)
    # else:
    #     norm_score = 0    
    
    # if good_listR[n] == True:
    # conv = Model_coef*norm_score
    # else: 
        # conv = Model_coef*0
    if np.mean(norm_score*norm_score*1e4) > weight_thresh*1e2:
        conv = Model_coef
    else:
        conv = Model_coef*0
    
    # norm_score_all[n,:] = norm_score.T
    for b_ind in np.arange(np.size(Model_coef, 0)):
        Convdata[b_ind][n, :] = conv[b_ind, :]


x_axis = np.arange(1, prestim+t_period, window)
fig, axes = plt.subplots(1,1,figsize = (10,8))

for f in range(ax_sz):
        error = np.std(Convdata[f],0)/np.sqrt(np.size(good_list_sep))
        y = ndimage.gaussian_filter(np.mean(Convdata[f],0),2)
        axes.plot(x_axis*1e-3-prestim*1e-3,y,c = cmap3[f],linestyle = lstyles[f])
        axes.fill_between(x_axis*1e-3-prestim*1e-3,y-error,y+error,facecolor = cmap3[f],alpha = 0.3)
        axes.set_ylim([-0.20,0.20])

# axes[1].plot(x_axis*1e-3-prestim*1e-3,ndimage.gaussian_filter(np.mean(norm_score_all,0),2))

e_lines = np.array([0, 500, 500+1000, 2500+1000])
e_lines = e_lines+500

# %% Calculating weights

weight = {}
p = {}
p[-1] = np.arange(140,160)
p[-2] = p[-1]
# p[-2] = np.arange(140,160)

    
for f in np.arange(ax_sz):  
    weight[-1,f]= np.zeros((1,294))
    weight[-2,f] = np.zeros((1,294))    
    for c_ind in [-1,-2]:
        if c_ind == -1:
            for n in np.arange(95,294):
                weight[c_ind,f][0,n] = np.mean(Convdata[f][n,p[c_ind]])
        if c_ind == -2:
            for n in np.arange(95):
                weight[c_ind,f][0,n] = np.mean(Convdata[f][n,p[c_ind]])
                
                
# fig, axes = plt.subplots(1,1,figsize = (10,8))
# axes.scatter(weight[-1,2],-weight[-2,2])
# axes.set_xlim([-0.5,0.5])
# axes.set_ylim([-0.5,0.5])

# %% 
weight = {}
nbunits = {}
for f in np.arange(ax_sz): 
    nbunits[-1,f] = np.zeros((1,7))
    nbunits[-2,f] = np.zeros((1,7))

for t in np.arange(7):
    p = np.arange(t*20,t*20+20)
    for f in np.arange(ax_sz):  
        weight[-1,f]= np.zeros((1,294))
        weight[-2,f] = np.zeros((1,294))

        for c_ind in [-1,-2]:
            if c_ind == -1:
                for n in np.arange(95,294):
                    weight[c_ind,f][0,n] = np.mean(Convdata[f][n,p])
                nbunits[c_ind,f][0,t] = np.sum((weight[c_ind,f] > 0.1))/200
    
            if c_ind == -2:
                for n in np.arange(95):
                    weight[c_ind,f][0,n] = np.mean(Convdata[f][n,p])
                nbunits[c_ind,f][0,t] = np.sum((weight[c_ind,f] > 0.1))/94



fig, axes = plt.subplots(4,1, figsize = (8,20))
for f in np.arange(ax_sz):
    axes[f].plot(np.arange(7), nbunits[-1,f].T, c = cmap[f], linestyle = "solid")
    axes[f].plot(np.arange(7), nbunits[-2,f].T, c = cmap[f], linestyle = "dotted")
    


                
    
















# %%

f = 2
list2 = (weight[-1,f] > 0.1) #* (weight[-2,f] == 0)
list3 = (weight[-2,f] > 0.1)# * (weight[-1,f] == 0)
print(np.sum(list2))
print(np.sum(list3))
list4 = (weight[-1,f] > 0)*(-weight[-2,f] > 0)
# result = stats.linregress(weight[-1,f][list4],-weight[-2,f][list4])

# print(result.rvalue)

# 
fig, axes = plt.subplots(1,1,figsize = (10,8))

# f = 1
# y1 = np.mean(np.concatenate((Convdata[f][list2[0,:],:], Convdata[f][list3[0,:],:])),0)
# s1 = np.std(np.concatenate((Convdata[f][list2[0,:],:], Convdata[f][list3[0,:],:])),0)/np.sqrt(np.sum(list2)+np.sum(list3))

y1 = np.mean(Convdata[f][list2[0,:],:],0)
s1 = np.std(Convdata[f][list2[0,:],:],0)/np.sqrt(np.sum(list2))
y2 = np.mean(Convdata[f][list3[0,:],:],0)
s2 = np.std(Convdata[f][list3[0,:],:],0)/np.sqrt(np.sum(list3))

# take history units

d = Convdata[f][list3[0,:],:]

# p1 = np.arange(90,110)
# t1 = Convdata[-1,f][list2[0,:],:][:,p1]

# t2 = Convdata[-2,f][list3[0,:],:][:,p1]
# stats.ks_2samp(np.mean(t1,1),np.mean(t2,1))

y1 = ndimage.gaussian_filter(y1,2)
y2 = ndimage.gaussian_filter(y2,2)

cmap = cmap3 = ['tab:purple','tab:blue','tab:red','tab:olive']

axes.plot(x_axis*1e-3-prestim*1e-3,y1,c = cmap[f],linestyle = 'solid')
axes.fill_between(x_axis*1e-3-prestim*1e-3,y1-s1,y1+s1,facecolor = cmap[f],alpha = 0.3)


axes.plot(x_axis*1e-3-prestim*1e-3,y2,c = cmap[f],linestyle = 'dashed')
axes.fill_between(x_axis*1e-3-prestim*1e-3,y2-s2,y2+s2,facecolor = cmap[f],alpha = 0.3)

# axes.set_ylim([-0.05,0.65])