# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:35:38 2022
        
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
    
    Y = np.zeros((N_trial,int(t_period/window)))
    for tr in range(N_trial):
        Y[tr,:] = D_ppc[n,0][0,D_ppc[n,2][tr,0]-1 - int(prestim/window): D_ppc[n,2][tr,0] + int(t_period/window)-1 - int(prestim/window)]
    


                
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
        l = L[:,w]*0
        # X2 = np.column_stack([np.ones_like(y),X[:,0],l,X[:,2:]])
        # X = np.column_stack([X[:,0],l,X[:,2:]])
        X3 = np.column_stack([l,X])
        
        # adding kernels to each task variable
        if w*window <= prestim-window:
            X3[:,1:4] = 0;
        elif w*window <= prestim+1500-window:
            
            if ca == 0:
                X3[:,3]= 0;
            elif ca == 1:
                for tr in np.arange(np.size(L,0)):
                    if np.isnan(Rt[tr,0]):
                        X3[tr,3] = 0;
                    else:
                        if w*window <= prestim + Rt[tr,0]*1e3 -window:
                            X3[tr,3] = 0;
                        
        
        Xm = np.zeros_like(X3)
        Xm[:,m_ind] = 1
        X3 = X3*Xm
        
        
        
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
            cmap = ['tab:orange','tab:purple','tab:blue','tab:red','tab:olive','tab:olive']
            clabels = ["lick","Contingency","stim","reward","history","history2"]
            lstyles = ['solid','solid','solid','solid','solid','dashed']
            # cmap = ['tab:orange','tab:purple','tab:purple','tab:blue','tab:blue','tab:red','tab:red']
            # clabels = ["lick","go","no-go","stim1","stim2","reward","history"]
            # lstyles = ['solid','solid','dashed','solid','dashed','solid','dashed']
            # cmap = ['tab:purple', 'tab:orange','tab:blue','tab:olive']
            # clabels = ["contin","action","stim","history"]
        # elif c_ind == 1 or c_ind ==2:
        #     cmap = ['tab:orange', 'tab:green','tab:blue']
        #     clabels = ["action","correct","stim"]        
        # else:     # c_ind == -3 or c_ind == -4      
        #     cmap = ['tab:orange', 'tab:green','tab:blue','tab:olive']
        #     clabels = ["action","correct","stim","history"]
            
            
            
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
        # stim_ind1 = X3[:,3] == 1     
        # stim_ind2 = X3[:,3] == 0  
        
    
        # ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind1,:],0),2),
        #           linewidth = 2.0, color = cmap[3],label = 'Reward',linestyle = lstyles[3])
        # ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind2,:],0),2),
        #           linewidth = 2.0, color = cmap[3],label = 'No Reward',linestyle = lstyles[4])
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

Nvar = 5

def build_model(n, t_period, prestim, window,k,c_ind,ca):
    for m_ind in np.arange(Nvar):
        X, Y, Yhat, Model_Theta, score, Yhat1, Yhat2 = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,m_ind,0)
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
                X, Y, Yhat, Model_Theta, score, Yhat1, Yhat2 = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,m_ind2,0)
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

Data = np.load('D:\Python\Data_PPCAll_Ca_05_26.npy',allow_pickle= True).item()

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
            # maxS = [0,1,2,3,4]   
            X, Y, Yhat, Model_Theta, score, Yhat1, Yhat2 = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,maxS,1)
            Data[n,c_ind-1] = {"X":X,"coef" : Model_Theta, "score" : score, 'Y' : Y,'Yhat' : Yhat,'maxS' : maxS}
            # t += 1
            # print(t,"/",len(good_list))
            good_list2 = np.concatenate((good_list2,[n]))
            
            
        except KeyboardInterrupt:
            
            break
        except:
            
            print("Error, probably not enough trials") 
# np.save('Data_PPCAll_Ca_05_26.npy', Data,allow_pickle= True)     

# %% Save Data since it takes forever
# # np.save('Data_PPCAll_Ca_v3.npy', Data)

# Data = np.load('D:\Python\Data_PPCAll_Ca_V3.npy',allow_pickle= True).item()
# score_diff = np.zeros((len(good_list),160))

# t = 0
# for n in good_list:
#     n = int(n)
    
#     score_2 = np.mean(Data[n,c_ind-1]["score"],1).T
#     score_2[score_2 <= 0] = 0
#     score_1 = np.mean(DataS[n,c_ind-1]["score"],1).T
#     score_1[score_1 <= 0] = 0
    
    
    
#     score_diff[t,:] = (score_2-score_1)/(max([max(score_1),max(score_2)]) +0.02)
#     t += 1
    
    
    
# fig, axs = plt.subplots(1,1,figsize= (10,8))

# x_axis = np.arange(160)*0.05-1


# s,p = stats.ttest_1samp(score_diff,np.zeros((1,160)))
# p = p> 0.05

# axs.plot(x_axis,ndimage.gaussian_filter(np.mean(score_diff,0), 1),linewidth = 2.0)
# axs.fill_between(x_axis,ndimage.gaussian_filter(np.mean(score_diff,0), 1)-ndimage.gaussian_filter(np.std(score_diff,0), 1),
#                  ndimage.gaussian_filter(np.mean(score_diff,0), 1)+ndimage.gaussian_filter(np.std(score_diff,0), 1)
#                  ,alpha = 0.2)


# T = x_axis*p
# T2 = np.ones_like(p)*0.05
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
        

# %% Error calculations

cat_list = best_kernel[c_ind][0,:] != 0 # Only neurons that were categorized

good_list_sep = good_list[cat_list]

E = np.zeros((2,np.size(good_list_sep,0)))

i= 0
for n in good_list_sep:
    n = int(n)
    E[0,i] = np.mean(np.abs(Data[n,c_ind-1]["Y"][0:200,:]-Data[n,c_ind-1]["Yhat"][0:200,:]))
    E[1,i] = np.mean(np.abs(Data[n,c_ind-1]["Y"][200:,:]-Data[n,c_ind-1]["Yhat"][200:,:]))
    i += 1

    
    
fig, axes = plt.subplots(1,1,figsize = (10,8))
axes.scatter(E[0,:], E[1,:])
# %% plot piechart for all trials

def pie_all_rules(best_kernel): 
    
    d_list = good_list > 179

    d_list3 = good_list <= 179
    
    # good_list_sep = np.int_(good_list[d_list])
    b_list = np.arange(np.size(good_list))
    b_list = b_list[:]

    
    

    if c_ind == -2:
        # pie_labels = ["Uncategorized", "lick","go","no-go","stim1","stim2","reward","history"]
        # cmap = ['tab:gray','tab:orange','tab:purple','tab:purple','tab:blue','tab:blue','tab:red','tab:olive']
        pie_labels = ["Uncategorized", "lick","Contingency","stim","reward","history"]
        cmap = ['tab:gray','tab:orange','tab:purple','tab:blue','tab:red','tab:olive']    
    plt.pie(np.bincount(best_kernel[c_ind][1,b_list].astype(int)),labels = pie_labels, colors = cmap)
    plt.show() 
    
    print(np.bincount(best_kernel[c_ind][1,b_list].astype(int)))

pie_all_rules(best_kernel)



# %% Accumulated task variable encoding piechart 04/17

def pie_accumulated(Data,best_kernel):
    d_list = good_list > 179

    d_list3 = good_list <= 179
    
    # good_list_sep = np.int_(good_list[d_list])
    b_list = np.arange(np.size(good_list))
    b_list = b_list[d_list3]
    pie_labels = [ "lick","Contingency","stim","reward","history"]
    cmap = ['tab:orange','tab:purple','tab:blue','tab:red','tab:olive'] 
    cat_concat = [];
    for n in b_list:
        if best_kernel[c_ind][1,n] > 0:
            try: 
                cat_concat = np.concatenate((cat_concat,Data[int(good_list[n]),c_ind-1]["maxS"]))
            except:
                cat_concat = np.concatenate((cat_concat,[Data[int(good_list[n]),c_ind-1]["maxS"]]))
                
    plt.pie(np.bincount(cat_concat.astype(int)),labels = pie_labels, colors = cmap)
    plt.show() 
    print(np.bincount(cat_concat.astype(int)))
      
pie_accumulated(Data,best_kernel)
        
       


    
    

# %% Normalized population average of task variable weights

d_list = good_list > 179

d_list3 = good_list <= 179

cat_list = best_kernel[c_ind][0,:] != 0 # Only neurons that were categorized

# good_list_sep = good_list[cat_list]
good_list_sep = good_list[:]

weight_thresh = 2*1e-2


if c_ind == 0 or c_ind == -2:
    cmap3 = ['tab:orange','tab:purple','tab:blue','tab:red','tab:olive']
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
    Model_coef = Model_coef/(np.max(np.abs(Model_coef)) + 0.1) # soft normalization value for model_coef

    
    norm_score = np.mean(Model_score, 1)
    norm_score[norm_score < weight_thresh] = 0
    if np.max(norm_score)>0:
        norm_score = norm_score/(np.max(norm_score)+weight_thresh)
    else:
        norm_score = 0    
    conv = Model_coef*norm_score
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
        axes.set_ylim([0,0.15])

# axes[1].plot(x_axis*1e-3-prestim*1e-3,ndimage.gaussian_filter(np.mean(norm_score_all,0),2))

e_lines = np.array([0, 500, 500+1000, 2500+1000])
e_lines = e_lines+500

# %% PCA
ax_sz = 5
tvlist = np.load('tvlist_PIC.npy',allow_pickle= True).item()

tvlist2= {}
tvlist2[4] = tvlist[3]
tvlist2[3] = tvlist[2]
tvlist2[2] = tvlist[1]
tvlist2[1] = tvlist[1]
tvlist2[0] = tvlist[0]

fig, axs = plt.subplots(ax_sz,6,figsize = (20,20))

d_list = good_list > 179

d_list3 = good_list <= 179


pca = {};
for f in np.arange(1,ax_sz):
    # pca[f] = SparsePCA(n_components=10,alpha = 0.01)  
    pca[f] = PCA(n_components=14) 
    # test = pca[f].fit_transform(ndimage.gaussian_filter(Convdata[f][:,:].T,[2,0])) # change to [2,0] if SU data, else, [1,0]
    test = pca[f].fit_transform(ndimage.gaussian_filter(Convdata[f][tvlist2[f][0],:].T,[1,0]))
    
    test = test.T
    for t in range(5):
        axs[f,t].plot(test[t,:],c = cmap3[f])
    axs[f,5].plot(np.cumsum(pca[f].explained_variance_ratio_[0:5]))
    plt.savefig("test.svg", format = 'svg')

    

#  Subspace overlap analysis
            
n_cv = 100

np.save('pca_common_all_PIC.npy',pca)

# p_list = {};
# # p_list[0] = good_list[d_list].astype(int)
# # p_list[1] = good_list[d_list3].astype(int)
# n_neuron = len(good_list[d_list])

# p_list[0] = np.arange(95)
# p_list[1] = np.arange(95,len(good_list))
# %% subspace overlap, angle method
def list_shuffle(n,m,fract):
    p_list = {};
    p_list[0] = np.arange(n)
    p_list[1] = np.arange(n,m)
    
    for p in [0,1]:
        lp = int(np.floor(n*fract))
        shuffle  = np.random.choice([True, False],n, p = [lp/n, 1-lp/n])
        
        if p == 0:
            test = np.where(shuffle == False)
            for pp in test[0]:
                p_list[p][pp] = np.random.choice(p_list[1],1)
        elif p == 1:
            test = np.where(shuffle == False)
            for pp in test[0]:
                p_list[p][pp] = np.random.choice(p_list[0],1)
    
    return p_list



Overlap = {};
Overlap[0] = np.zeros((ax_sz,ax_sz,n_cv)); # PPC_IC
Overlap[1] = np.zeros((ax_sz,ax_sz,n_cv)); # PPC_AC
Overlap_across = np.zeros((ax_sz,ax_sz,n_cv));

O_mean = {}
O_std = {}
O_mean[0] = np.zeros((ax_sz,ax_sz));
O_std[0] = np.zeros((ax_sz,ax_sz));
O_mean[1] = np.zeros((ax_sz,ax_sz));
O_std[1] = np.zeros((ax_sz,ax_sz));


n_list = {};
n_list[0] = np.arange(95)
n_list[1] = np.arange(95,len(good_list))

k1 = 0
k2 = 19


for f in np.arange(ax_sz):
    for f2 in np.arange(ax_sz):
        for k in np.arange(n_cv):
            p_list = list_shuffle(95, len(good_list), 0.9)

            for p in [0,1]: # PPC_IC and PPC_AC
                S_value = np.zeros((1,20))
                
                for d in np.arange(0,20):
                    # S_value2 = np.zeros((1,20))
                    # for d2 in np.arange(0,20):
                    S_value[0,d] = np.abs(np.dot(pca[f].components_[d,p_list[p]], pca[f2].components_[d,n_list[p]].T))
                        
                    # d2 = np.argmax(S_value2)
                    S_value[0,d] = S_value[0,d]/(np.linalg.norm(pca[f].components_[d,p_list[p]])*np.linalg.norm(pca[f2].components_[d,n_list[p]]))
                        
                Overlap[p][f,f2,k] = np.max(S_value)
            # Overlap_across[f,f2,k] = np.max(np.abs(np.dot(f].components_[:,n_ind[0]], pca[c_list[1],f2].components_[:,n_ind[1]].T)*np.identity(20)))
        for p in [0,1]:
            O_mean[p][f,f2] = np.mean(Overlap[p][f,f2,:])
            O_std[p][f,f2] = np.std(Overlap[p][f,f2,:])


O_mean2 = {}
O_std2 = {}
O_mean2[0] = np.zeros((ax_sz,ax_sz));
O_std2[0] = np.zeros((ax_sz,ax_sz));
O_mean2[1] = np.zeros((ax_sz,ax_sz));
O_std2[1] = np.zeros((ax_sz,ax_sz));

for f in np.arange(ax_sz):
    for f2 in np.arange(ax_sz):
        for p in [0,1]:
            O_mean2[p][f,f2] = np.mean([O_mean[p][f,f2],O_mean[p][f2,f]])
            O_std2[p][f,f2] = np.mean([O_std[p][f,f2],O_std[p][f2,f]])
            
#         cmap3 = ['tab:orange','tab:purple','tab:blue','tab:red','tab:olive']    
# for p in [0,1]:            
#     fig, axes = plt.subplots(ax_sz,1, figsize =(7, 20))
#     for f in np.arange(ax_sz):
#         axes[f].bar(np.arange(ax_sz), O_mean2[p][f,:], yerr = O_std2[p][f,:], color = cmap3)
        

# fig, axes = plt.subplots(1,1,figsize = (10,8))
# for p in [0,1]:
#     axes.bar( [p, 2+p,4+p], O_mean2[p][0,[0,1,3]], yerr = O_std2[p][0,[0,1,3]], color = ['tab:orange','tab:purple','tab:red'])
#     axes.set_ylim([0,1])


# # x1 = [.8,1.8,2.8]
# # y1 = [O_mean[0,0],O_mean[1,1],O_mean[2,2]]
# # e1 = [O_std[0,0],O_std[1,1],O_std[2,2]]

# test1 = Overlap[0][1,3,:]
# test2 = Overlap[1][1,3,:]
# stats.ks_2samp(test1,test2)





# %% Calculate var explained percentage by PC 4-20 for R

# run R and PCA with separate subpopulations

d_list1 = good_list < 179
d_list2 = good_list > 179


R = ndimage.gaussian_filter(Convdata[4][d_list2,:].T,[1,0])
R0 = R[0:20,:]
R = R[40:,:]

npc = [0,3,20] # pc 1 to 3, 4 to 20 

V = 1-np.linalg.norm(R0[:,:] - np.dot(np.dot(R0[:,:],pca[0].components_[npc[1]:npc[2],:].T),
                                                        pca[0].components_[npc[1]:npc[2],:]))/np.linalg.norm(R0[:,:])


# V_ac = 1-np.linalg.norm(R[:,d_list2] - np.dot(np.dot(R[:,d_list2],pca[4].components_[npc[0]:npc[1],d_list2].T),
#                                                         pca[4].components_[npc[0]:npc[1],d_list2]))/np.linalg.norm(R[:,d_list2])

# V_ic = 1-np.linalg.norm(R[:,d_list1] - np.dot(np.dot(R[:,d_list1],pca[4].components_[npc[0]:npc[1],d_list1].T),
#                                                         pca[4].components_[npc[0]:npc[1],d_list1]))/np.linalg.norm(R[:,d_list1])



Vp = np.zeros((2,ax_sz,20))

for cv in np.arange(20):
    d_list1 = good_list < 179
    d_list2 = good_list > 179
 
    for s in np.arange(np.size(good_list)):
        if d_list1[s] == True:
            shuffle = np.random.choice(2,1, p = [0.8,0.2])
            if shuffle == 1:
                d_list1[s] = False
        
        if d_list2[s] == True:
            shuffle = np.random.choice(2,1, p = [0.8,0.2])
            if shuffle == 1:
                d_list2[s] = False
                
    # R2 = R[:,d_list1]
    # R3 = R[:,d_list2]               
        
    for f in np.arange(ax_sz):
        # V1 = 1-np.linalg.norm(R2 - np.dot(np.dot(R2,pca[f].components_[npc[0]:npc[1],d_list1].T),
        #                                                     pca[f].components_[npc[0]:npc[1],d_list1]))/np.linalg.norm(R2)
        V2 = 1-np.linalg.norm(R0 - np.dot(np.dot(R0,pca[f].components_[npc[0]:npc[2],:].T),
                                                            pca[f].components_[npc[0]:npc[2],:]))/np.linalg.norm(R0)
        
        # Vp[0,f,cv] = V1/V_ic
        Vp[1,f,cv] = V2 #/V


    
Vpmean = np.mean(Vp,axis = 2)
Vperr = np.std(Vp,axis = 2)

fig, axes = plt.subplots(1,1, figsize = (10,8))

for f in np.arange(ax_sz):
    axes.bar(f, Vpmean[1,f],yerr = Vperr[1,f],color = cmap3[f])


# axes.set_ylim([0,0.7])   

# fig, axes = plt.subplots(1,1, figsize =(10, 8))
# for p in [0,1]:
#     for f in np.arange(ax_sz-1):
#         axes.bar(f+ p*4, Vpmean[p,f],yerr = Vperr[p,f],color = cmap3[f])



# %% dendrogram

from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(O_mean2[0],'single')

dendrogram(Z,labels = clabels)

# %% Calculate explained variance by each subspace across time
array_length = np.size(Convdata[0],1)

xtime = np.arange(array_length)*50*1e-3-prestim*1e-3

n_pc = 3
n_pc1 = 0
n_cv = 20;
d_list1 = good_list > 179
d_list2 = good_list < 179
V_cap1  =np.zeros((ax_sz,array_length,n_cv))
V_cap2  =np.zeros((ax_sz,array_length,n_cv))

V_cap1_base = np.zeros((ax_sz,n_cv))
V_cap2_base = np.zeros((ax_sz,n_cv))

fig, axes = plt.subplots(ax_sz,1,figsize = (5,10))
for f  in np.arange(ax_sz): 
    
    
    
    
    R = ndimage.gaussian_filter(Convdata[f].T,[1,0])
    
    for cv in np.arange(20):
        r_shuffle = np.arange(len(good_list))
        np.random.shuffle(r_shuffle)
        R2 = R[:,r_shuffle]
        V_cap1_base[f,cv] = 1-np.linalg.norm(R2 - np.dot(np.dot(R2,pca[f].components_[n_pc1:n_pc,:].T),
                                                                pca[f].components_[n_pc1:n_pc,:]))/np.linalg.norm(R2)
        
                
        d_list1 = good_list > 179
        
        d_list2 = good_list <= 179
            # 20% shuffle
    
        for s in np.arange(np.size(good_list)):
            if d_list1[s] == True:
                shuffle = np.random.choice(2,1, p = [0.8,0.20])
                if shuffle == 1:
                    d_list1[s] = False
            
            if d_list2[s] == True:
                shuffle = np.random.choice(2,1, p = [0.8,0.20])
                if shuffle == 1:
                    d_list2[s] = False
     
        # d_list1 = good_list >0
        # d_list1 = good_list >0
        # d_list2 = good_list >0
        # V_cap2_base[f,cv] = 1-np.linalg.norm(R[:,d_list2] - np.dot(np.dot(R[:,d_list2],
        #                                                       pca[f].components_[:,d_list2].T),
        #                                                         pca[f].components_[:,d_list2]))/np.linalg.norm(R[:,d_list2])
        
        for t in np.arange(array_length):    
            V_cap1[f,t,cv] = 1-np.linalg.norm(R[t,d_list1] - np.dot(np.dot(R[t,d_list1],
                                                                  pca[f].components_[n_pc1:n_pc,d_list1].T),
                                                                    pca[f].components_[n_pc1:n_pc,d_list1]))/np.linalg.norm(R[t,d_list1])
            V_cap2[f,t,cv] = 1-np.linalg.norm(R[t,d_list2] - np.dot(np.dot(R[t,d_list2],
                                                                  pca[f].components_[n_pc1:n_pc,d_list2].T),
                                                                    pca[f].components_[n_pc1:n_pc,d_list2]))/np.linalg.norm(R[t,d_list2])
    
        
    
    
    tot_var = np.zeros((1,array_length))
    for t in np.arange(array_length):
        tot_var[0,t] = np.var(R[t,d_list])
        
    plt.plot(xtime,tot_var.T)
        
    
    axes[f].plot(xtime, np.mean(V_cap1[f,:,:],1),c = cmap3[f],linestyle = 'solid')
    axes[f].fill_between(xtime,np.mean(V_cap1[f,:,:],1)- np.std(V_cap1[f,:,:],1),np.mean(V_cap1[f,:,:],1)+ np.std(V_cap1[f,:,:],1),facecolor = cmap3[f],alpha = 0.2)

    axes[f].plot(xtime, np.mean(V_cap2[f,:,:],1),c = cmap3[f],linestyle = 'dotted')
    axes[f].fill_between(xtime,np.mean(V_cap2[f,:,:],1)- np.std(V_cap2[f,:,:],1),np.mean(V_cap2[f,:,:],1)+ np.std(V_cap2[f,:,:],1),facecolor = cmap3[f],alpha = 0.2)


    y =np.mean(V_cap1_base[f,:])
    error = np.std(V_cap1_base[f,:])
    axes[f].hlines(y, 
              xmin = min(xtime), 
              xmax = max(xtime),
              linestyles = 'dashed',
              colors = 'black', 
              linewidth = 2.0)
    # axes[f].fill_between(np.arange(110),y-error,y+error,facecolor = 'black',alpha = 0.3)

# %% PCA angle difference?

# # ref_ind = 2
# # comp_ind = 0
# # angle1 = np.dot(pca_all[4].components_[ref_ind,d_list], pca_AC[4].components_[comp_ind,:])
# # angle2 = np.dot(pca_all[4].components_[ref_ind,d_list2], pca_IC[4].components_[comp_ind,:])

# A = {}
# for f in np.arange(ax_sz):
#     A[f] = np.zeros((2,4))
#     for ref_ind in [0,1]:
#         for comp_ind in [0,1]:
#             A[f][ref_ind,comp_ind] = np.dot(pca_all[f].components_[ref_ind,d_list], pca_AC[f].components_[comp_ind,:])
#             A[f][ref_ind,comp_ind+2] = np.dot(pca_all[f].components_[ref_ind,d_list2], pca_IC[f].components_[comp_ind,:])
#     A[f] = np.abs(A[f])
    
    
    
# %% PCA Trajectories



from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
d_list1 = good_list > 179

d_list3 = good_list <= 179


def draw_traj(traj,f,v):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    styles = ['solid', 'solid', 'solid','dotted']
    cmap_names = ['summer','autumn','winter','summer']
    for tr in [0,1,2,3]:
        x = traj[f][tr][:,0]
        y = traj[f][tr][:,1]
        z = traj[f][tr][:,2]
        if ca == 0:
            x = ndimage.gaussian_filter(x,1)
            y = ndimage.gaussian_filter(y,1)
            z = ndimage.gaussian_filter(z,1)            
            
        time = np.arange(len(x))
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
        
    
        
        # norm = plt.Normalize(time.min(), time.max())
        cmap=plt.get_cmap(cmap_names[tr])
        # colors=[cmap(float(ii)/(n-1)) for ii in range(np.size(segments,0))]
        
        
        norm = BoundaryNorm([0,19,29,49,89,109],cmap.N)
        lc = Line3DCollection(segments, cmap=cmap_names[tr], norm=norm,linestyle = styles[0])
        lc.set_array(time)
        lc.set_linewidth(2)
        ax.add_collection3d(lc)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        if tr ==0:
            ax.auto_scale_xyz(x,y,z)
        # fig.colorbar(line, ax=axs[0])
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')
    # gif_filename = 'trajectory'
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
            plt.savefig('Images/img' + str(n).zfill(3) + '.png',
                        bbox_inches='tight')
    
# %%

plt.close()
draw_traj(traj,4,1)
import IPython.display as IPdisplay
import glob
from PIL import Image as PIL_Image
import imageio

images = [PIL_Image.open(image) for image in glob.glob('images/*.png')]
file_path_name = 'images/GLM_kernel/SU_trajectory_History.gif'
imageio.mimsave(file_path_name, images)

# writeGif(file_path_name, images, duration=0.1)
# IPdisplay.Image(url=file_path_name)
# %%

fig, axes = plt.subplots(ax_sz,2,figsize = (5,10))
traj = {}
xtime = np.arange(160)*5*1e-3-1e3
for f in np.arange(ax_sz):
    R = ndimage.gaussian_filter(Convdata[f].T,[1,0])
    
    traj[f] = {}
    # traj[f][0] = pca[f].fit_transform(R)
    traj[f][0] = np.dot(R,pca[f].components_.T)  
    traj[f][1] = np.dot(R[:,d_list1], pca[f].components_[:,d_list1].T) #*(len(good_list)/np.sum(d_list1))
    traj[f][2] = np.dot(R[:,d_list3], pca[f].components_[:,d_list3].T) #*(len(good_list)/np.sum(d_list3))
    traj[f][3] = traj[f][1] + traj[f][2]  

                           
    # traj[f][1] = np.dot(R,pca[f].components_.T)      # np.dot(R[:,d_list1], pca[f].components_[:,d_list1].T) #*(len(good_list)/np.sum(d_list1))
    # traj[f][2] = np.dot(R,pca[f].components_.T)      # np.dot(R[:,d_list3], pca[f].components_[:,d_list3].T) #*(len(good_list)/np.sum(d_list3))
    # traj[f][3] = np.dot(R,pca[f].components_.T)      # traj[f][1] + traj[f][2]

    draw_traj(traj,f,0)
    distance = {}
    distance[0] = np.linalg.norm(traj[f][0][:,0:3]-traj[f][1][:,0:3],axis = 1)
    distance[1] = np.linalg.norm(traj[f][0][:,0:3]-traj[f][2][:,0:3],axis = 1)
    distance[2] = np.linalg.norm(traj[f][1][:,0:3]-traj[f][2][:,0:3],axis = 1)
     
    
    axes[f,0].plot(xtime,distance[0], linestyle = 'solid')
    axes[f,0].plot(xtime,distance[1], linestyle = 'dashed')
    
    axes[f,1].plot(xtime,distance[2])
 
    # axes[f,1].plot(xtime,distance[0]-(distance[0]+distance[1])/2, linestyle = 'solid')
    # axes[f,1].plot(xtime,distance[1]-(distance[0]+distance[1])/2, linestyle = 'dashed')   
    





# %% Archived code, save for later 

# # %% plotting beta weights of all significant neurons 


# fig, axes = plt.subplots(4,2,figsize = (10,10))
# bins = np.arange(1,20)
# # cmap3 = ['tab:orange','tab:green','tab:blue',]
# cmap3 = ['tab:purple','tab:orange','tab:green','tab:blue']
# x_axis = bins*0.250
# # for f in range(3):
# #     axes[f,0].scatter(best_kernel[1][0,:],best_kernel[1][f+2,:],c = cmap3[f])
# #     axes[f,1].scatter(best_kernel[2][0,:],best_kernel[2][f+2,:],c = cmap3[f])
# #     axes[f,0].set_ylim([0,1])
# #     axes[f,1].set_ylim([0,1])
# #     axes[f,0].set_xticks(bins[1::2], x_axis[1::2])
# #     axes[f,1].set_xticks(bins[1::2], x_axis[1::2])

# for f in range(4):
#     axes[f,0].scatter(best_kernel[0][0,:],best_kernel[0][f+2,:],c = cmap3[f])
#     # axes[f,1].scatter(best_kernel[2][0,:],best_kernel[2][f+2,:],c = cmap3[f])
#     axes[f,0].set_ylim([0,1])
#     # axes[f,1].set_ylim([0,1])
#     axes[f,0].set_xticks(bins[1::2], x_axis[1::2])
#     # axes[f,1].set_xticks(bins[1::2], x_axis[1::2])
    
# #%% Plotting boxplot for each timewindow
# beta_time = {}

# for c_ind in [1,2]:
#     for f in range(3):
#         beta_time[f,c_ind] = [];
#         weight_thresh = 0.1
#         for b in bins:
#             ind = best_kernel[c_ind][0,:] == b
#             ind2 = best_kernel[c_ind][f+2,ind] > weight_thresh
#             beta_time[f,c_ind].append(best_kernel[c_ind][f+2,ind][ind2])


# fig, axes = plt.subplots(3,2,figsize = (10,10))


# for c_ind in [1,2]:
#     for f in range(3):
#         medianprops = dict(linestyle='-.', linewidth=2.5, color=cmap3[f])
#         axes[f,c_ind-1].boxplot(beta_time[f,c_ind],medianprops= medianprops)
#         axes[f,c_ind-1].set_xticks(bins[1::2], x_axis[1::2])
#         axes[f,c_ind-1].set_ylim(0,1)


# axes[0,0].set_title('Rule1')
# axes[0,1].set_title('Rule2')

# # plt.hist(best_kernel[1][0,:],bins)


# sizeX = 0
# p = 0

# for n in good_list:
#     n = int(n)
#     X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
#     if sizeX != np.size(X,0):        
#         fig, ax = plt.subplots(1,1,figsize = (5,5))
#         ax.plot(np.mean(L,0)*20)
#         sizeX = np.size(X,0)
#         p += 1
        

hist = np.zeros((1,294))
for n in np.arange(294):
    gn = good_list[n]
    if best_kernel[-2][1,n] > 0:
        hist[0,n] = np.size(Data[int(gn),c_ind-1]["maxS"])


h, bins = np.histogram(hist)
plt.hist(bins[:-1], bins, weights=h)

np.bincount(hist)
plt.hist(hist,np.arange(6))

    


