# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:04:29 2023

@author: Jong Hoon Lee
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:21:02 2023
        
    Code c_ind:
         1   =   Rule 1, without Trial History
        -1   =   Rule 1, with Trial History
         2   =   Rule 2, without Trial History
        -2   =   Rule 2, without Trial History   
        
        
   structure of task variable matrix X:      
       (trial,1) = stim onset time
       (trial,2) = rule (Rule 1 or 2)
       (trial,3) = contingency (1: go stim, 0: nogo stim)
       (trial,4) = Lick (1: lick, 0: No lick)
       (trial,5) = Correct choice (1: correct, 0: incorrect)
       (trial,6) = stage info (1: Task, 0: Conditioning)
       


@author: Jong Hoon Lee
"""

""" 
    Separating Rule1 vs Rule2 from main code.
    Analysis becomes different, it's starting to make less sense to keep
    both codes in one.

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


# %% File name and directory

# change fname for filename

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
            Y[tr,:] = D_ppc[n,0][0,D_ppc[n,2][tr,0]-1 
                                 - int(prestim/window): D_ppc[n,2][tr,0] 
                                 + int(t_period/window)-1 - int(prestim/window)]
        if np.mean(Y) > 0.5:
            good_list = np.concatenate((good_list,[n]))
    
    
    return good_list


@jit(target_backend='cuda')                         
def import_data_w_spikes(n,prestim,t_period,window,c_ind):
    D_ppc = load_matfile()
    S_all = np.zeros((1,max(D_ppc[n,2][:,0])+t_period+100))
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
    
    
    X = D_ppc[n,2][:,2:6] # task variables
    Y = [];
    S = np.concatenate((S_pre,S),1)
    t_period = t_period+prestim
    
    
    if c_ind !=3:
    # remove conditioning trials     
        S = np.concatenate((S[0:200,:],S[D_ppc[n,5][0][0]:,:]),0)
        X = np.concatenate((X[0:200,:],X[D_ppc[n,5][0][0]:,:]),0)

    # only contain conditioningt trials
    else:
        S = S[201:D_ppc[n,5][0][0]]
        X = X[201:D_ppc[n,5][0][0]]


    N_trial2 = np.size(S,0)

    # select analysis and model parameters with c_ind    
    
    if c_ind ==1 or c_ind ==-1: # Rule 1
        r_ind = np.arange(200)
    elif c_ind ==2 or c_ind ==-2: # Rule 2
        r_ind = np.arange(200,np.size(X,0))
        
        
    # Adding previous trial correct vs wrong
    XHist = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
    XHist = XHist[:,None]
    X = np.concatenate((X,XHist),1) # History is added at the end
      
    for w in range(int(t_period/window)):
        y = np.mean(S[:,range(window*w,window*(w+1))],1)*1e3
        Y = np.concatenate((Y,y))
        
    Y = np.reshape(Y,(int(t_period/window),N_trial2)).T
    
    Y = Y[r_ind,:]
    X = X[r_ind,:]
    X = X[:,[1,3,4]] # removing contingency and correct
        

    return X, Y

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

    


    if c_ind ==1 or c_ind ==-1: # Rule 1
        r_ind = np.arange(200)
    elif c_ind ==2 or c_ind ==-2: # Rule 2
        r_ind = np.arange(D_ppc[n,4][0][0],np.size(X,0))
        
    
        
    # Add reward  history
    Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
    Xpre = Xpre[:,None]
    X2 = np.column_stack([X[:,3],
                         X[:,2]*X[:,1],Xpre]) 
    # Add reward instead of action

    
    # # Adding previous trial correct vs wrong
    # XHist = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
    # XHist = XHist[:,None]
    # X = np.concatenate((X,XHist),1) # History is added at the end
        
    Y = Y[r_ind,:]
    # normalize Y 
    # Y = ndimage.gaussian_filter(Y,[1,0])
    # Y = Y/(np.max(ndimage.gaussian_filter(Y.T,[0,1]))+0.5)
    # Y = Y/(np.max(Y)+0.5)
    X2 = X2[r_ind,:]
    L2 = L2[r_ind,:]

    # X = X[:,[1,3,4]] # removing contingency and correct
        
        
    return X2,Y, L2, Rt 


# plt.plot(ndimage.gaussian_filter(Y.T,[0,1]))

# %% glm_per_neuron function code.
# Main functions start here. 

def glm_per_neuron(n,t_period,prestim,window,k,c_ind,ca,m_ind,fig_on): 
    # if using spike data
    if ca == 0:
        X, Y, Y2,L = import_data_w_spikes(n,prestim,t_period,window,c_ind)
    else:
    # if using Ca data
        X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
        Y2 = Y
    
    
    t_period = t_period+prestim
    Yhat = [];
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
        l = L[:,w]
        # X2 = np.column_stack([np.ones_like(y),X[:,0],l,X[:,2:]])
        # X = np.column_stack([X[:,0],l,X[:,2:]])
        # X10 = X[:,0]*-1 +1
        
        # X3 = np.column_stack([l,X])
        X3 = X
        # X3 = np.column_stack([X10,X])
        X4  = X3
        Xm = np.zeros_like(X3)
        Xm[:,m_ind] = 1
        X3 = X3*Xm

        # if w*window > prestim + window:
        #     X3[:,3] = 0;
        # adding kernels to each task variable
        if w*window <= prestim-window:
            X3[:,0:2] = 0;
        elif w*window <= prestim+1500-window:
            
            if ca == 0:
                X3[:,1]= 0;
            elif ca == 1:
                for tr in np.arange(np.size(L,0)):
                    if np.isnan(Rt[tr,0]):
                        X3[tr,1] = 0;
                    else:
                        if w*window <= prestim + Rt[tr,0]*1e3 -window:
                            X3[tr,1] = 0;
                        
        
        
        X2 = np.column_stack([np.ones_like(y),X3])
        ss= ShuffleSplit(n_splits=k, test_size=0.20, random_state=0)
        y2 = ndimage.gaussian_filter(y,0)
        
        # X3[X3[:,1] == 0,1] = -1
        cv_results = cross_validate(reg, X3, y2, cv = ss , 
                                    return_estimator = True, 
                                    scoring = 'r2')
        theta = np.zeros((np.size(X2,1)-1,k))
        inter = np.zeros((1,k))
        pp = 0
        for model in cv_results['estimator']:
            theta[:,pp] = model.coef_ 
            inter[:,pp] = model.intercept_
            pp = pp+1
        theta3 = np.concatenate((np.mean(inter,1),np.mean(theta,1)))
        yhat = X2 @ theta3
        
        
        
        score = np.concatenate((score, cv_results['test_score']))
        TT2 = np.concatenate((TT2,np.mean(theta,1)))
        Intercept = np.concatenate((Intercept,np.mean(inter,1)))
        CI2 = np.concatenate((CI2,stats.sem(theta,1)))

        Yhat = np.concatenate((Yhat,yhat))
        
    Yhat = np.reshape(Yhat,(int(t_period/window),N_trial2)).T

    
    TT2 = np.reshape(TT2,(int(t_period/window),np.size(X3,1))).T
    CI2 = np.reshape(CI2,(int(t_period/window),np.size(X3,1))).T
    score = np.reshape(score,(int(t_period/window),k))
    
    
    
    
    # Figures
    if fig_on == 1:
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))        
    
        if c_ind == 1 or c_ind ==2:
            cmap = ['tab:orange','tab:blue']
            clabels = ["action","stim"]        
        elif c_ind == -1 or c_ind == -2:     # c_ind == -3 or c_ind == -4      
            # cmap = ['tab:orange','tab:blue','tab:red','tab:olive']
            # clabels = ["lick","stim","Reward","history"]
            # lstyles = ['solid','solid','solid','dashed']
            cmap = ['tab:blue','tab:red','tab:orange']
            clabels = ["stim","Reward","history"]
            lstyles = ['solid','solid','solid']
            
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
        stim_ind1 = X4[:,1] == 1     
        stim_ind2 = X4[:,1] == 0  
        
    
        # ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind1,:],0),2),
        #          linewidth = 2.0, color = cmap[2],label = '5 kHz',linestyle = lstyles[2])
        # ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind2,:],0),2),
        #          linewidth = 2.0, color = cmap[2],label = '10 kHz',linestyle = lstyles[3])
        # ax1.set_title('Firing rate y')
        # ax1.legend(loc = 'upper right')
    
        
        # ax3.plot(x_axis,ndimage.gaussian_filter(np.mean(Yhat[stim_ind1,:],0),2),
        #          linewidth = 2.0, color = cmap[2],linestyle = lstyles[2])
        # ax3.plot(x_axis,ndimage.gaussian_filter(np.mean(Yhat[stim_ind2,:],0),2),
        #          linewidth = 2.0, color = cmap[2],linestyle = lstyles[3]) 
        # ax3.set_title('Prediction y_hat')
    
        ax2.set_title('unit_'+str(n+1))
        ax4.set_title('explained variance')
        ax4.set_ylim(bottom = -2, top = var_top)
        plt.show()
    
    
    Model_Theta = TT2

    return X3, Y, Yhat, Model_Theta, score, Intercept    



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

# %% Build model
Nvar = 3
def build_model(n, t_period, prestim, window,k,c_ind,ca):
    for m_ind in np.arange(Nvar):
        X, Y, Yhat, Model_Theta, score, inter = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,m_ind,0)
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
                X, Y, Yhat, Model_Theta, score, inter = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,m_ind2,0)
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
    0 : lick vs no lick
    1 : correct vs wrong
    2 : stim 1 vs stim 2
    3 : if exists, would be correct history (previous correct ) 

"""



t_period = 7000
prestim = 1000

window = 50 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 10 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [-1,-2]



if ca ==0:
    D_ppc = load_matfile()
    good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    good_list = find_good_data_Ca(t_period)

# %%
Data = {}

# additional code for explained variance comparison
DataS = {}
ax_sz = 3
S = np.zeros((1,ax_sz))
ana_period = np.array([0, t_period+prestim])
weight_thresh = 2*1e-2

# change c_ind and n here. 
# good_list3 = {}
# for c_ind in c_list:
#     t = 0 
#     good_list2 = [];
    # good_list3[c_ind] =[];
# good_list = np.arange(np.size(D_ppc,0))
for n in np.arange(np.size(D_ppc,0)): # good_list:
    for c_ind in c_list:
        n = int(n)
        # X, Y, Yhat, Model_Theta, score = glm_per_neuron(n, t_period, prestim, window,k,c_ind)
        # Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score} 
        try:
            
            maxS = build_model(n, t_period, prestim, window, k, c_ind, ca)
            # maxS = [0,1,2,3]  
            X, Y, Yhat, Model_Theta, score, intercept = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,maxS,1)
            Data[n,c_ind-1] = {"X" : X,"coef" : Model_Theta, "intercept" : intercept, "score" : score, 'Y' : Y,'Yhat' : Yhat, 'maxS' : maxS}


            # good_list2 = np.concatenate((good_list2,[n]))
            # print(t,"/",len(good_list))
            # if np.mean(np.abs(X[:,1]-X[:,2]),axis = 0) <= 0.99 and np.mean(np.abs(X[:,1]-X[:,2]),axis = 0) >= 0.01:
            #     good_list3[c_ind] = np.concatenate((good_list3[c_ind],[n]))
            
            print(n)
        except KeyboardInterrupt:
            break
        except:
            print("Error, probably not enough trials") 

# np.save('R1vsR2Data_norm.npy',Data, allow_pickle = True)

# Data = np.load('R1vsR2Data.npy',allow_pickle= True).item()
# %% testing model weight stuff

fig, axes = plt.subplots(2,2,figsize = (10,8))


x_axis = np.arange(1, prestim+t_period, window)
axes[0,0].plot(x_axis,np.mean(Y[X[:,1]==1,:],0), linestyle = 'solid')
axes[0,0].plot(x_axis,np.mean(Y[X[:,1]==0,:],0), linestyle = 'dotted')
axes[1,0].plot(x_axis,np.mean(Yhat[X[:,1]==1,:],0), linestyle = 'solid')
axes[1,0].plot(x_axis,np.mean(Yhat[X[:,1]==0,:],0), linestyle = 'dotted')

ymean = np.zeros((1,160))
ymean[0,:] = intercept

theta3 = np.concatenate((ymean,Model_Theta),0)
X2 = np.concatenate((np.ones((200,1)),X),1)

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
d_list = good_list > 179

d_list3 = good_list <= 179

good_list_sep = good_list[d_list]

Rscore = {}
for c_ind in c_list:
    Rscore[c_ind] = np.zeros((ax_sz+1,np.size(good_list)))
    
y_lens = np.arange(160)
for c_ind in [-1]:    
    for n in np.arange(np.size(good_list_sep,0)):
        # print(n)
        nn = good_list_sep[n]
        nn = int(nn)
        maxS = Data[nn,c_ind-1]["maxS"]
        try:
            X = Data[nn,c_ind-1]["X"]
        except:                
            X, Y, Yhat, Model_Theta, score, intercept = glm_per_neuron(nn, t_period, prestim, window,k,c_ind,ca,maxS,1)
            Data[nn,c_ind-1] = {"X" : X,"coef" : Model_Theta, "intercept" : intercept, "score" : score, 'Y' : Y,'Yhat' : Yhat, 'maxS' : maxS}
        
        Y = Data[nn,c_ind-1]["Y"][:,y_lens]
        Yhat = Data[nn,c_ind-1]["Yhat"][:,y_lens]
        # intercept = Data[nn,c_ind-1]["intercept"]
        Model_Theta = Data[nn,c_ind-1]["coef"]
        ymean = np.ones((len(y_lens),np.size(X,0))).T*Data[nn,c_ind-1]["intercept"][y_lens]
        # ymean[0,:] = intercept
        
        theta3 = np.concatenate(([ymean[0,:]],Model_Theta[:,y_lens]),0)
        X2 = np.concatenate((np.ones((np.size(X,0),1)),X),1)
        
        for f in np.arange(ax_sz):
            yhat2 = X2[:,[0,f+1]] @ theta3[[0,f+1],:]
            Rscore[c_ind][f,n] = 1- np.sum(np.square(Y-yhat2))/np.sum(np.square(Y-ymean))
            if Rscore[c_ind][f,n] ==0:
                Rscore[c_ind][f,n] = -1
                
        Rscore[c_ind][ax_sz,n] = 1- np.sum(np.square(Y-Yhat))/np.sum(np.square(Y-ymean))
        # Rscore[c_ind][:,n]    

# scatter_ind = [np.arange(ax_sz+1)]*np.ones((ax_sz+1,len(good_list))).T
# scatter_ind = scatter_ind.T

# %% 
cmap = ['tab:orange','tab:blue','tab:red','tab:olive']
c_ind = -1

d_list = good_list > 179
# 
d_list3 = good_list <= 179
# d_list3 = good_list > 179


def make_RS(d_list):
    fig, axes = plt.subplots(1,1, figsize = (10,8))
    Rsstat = {}
    for c_ind in c_list:
        for f in np.arange(0,ax_sz):
            Rs = Rscore[c_ind][f,d_list]
            Rmax = Rscore[c_ind][4,d_list]
            Rmax = Rmax[Rs>0.01]
            # Rs = Rs[Rs>0.01]
    
            # Rs = Rs/(Rmax+0.03)
            Rsstat[c_ind,f] = Rs
            axes.scatter(np.ones_like(Rs)*(f+(c_ind+1)*-0.3),Rs,c = cmap[f])
            axes.scatter([(f+(c_ind+1)*-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')
            # axes.boxplot(Rs,positions= [f+(c_ind+1)*-0.3])
        axes.scatter(np.ones_like(Rscore[c_ind][4,d_list])*(4+(c_ind+1)*-0.3),Rscore[c_ind][4,d_list])
        axes.scatter([(4+(c_ind+1)*-0.3)],np.mean(Rscore[c_ind][4,d_list]),c = 'k',s = 500, marker='_')
        
        Rsstat[c_ind,4] = Rscore[c_ind][4,d_list]
    
        # axes.boxplot(Rscore[c_ind][4,d_list3],positions= [4+(c_ind+1)*-0.3])
        axes.set_ylim([-0.05,1.0])

    return Rsstat

RsStat_PIC = make_RS(d_list3)
RsStat_PAC = make_RS(d_list)

    # RsStat_PIC = Rsstat

res = {}


f = 1
# y1 = np.concatenate((RsStat_PAC[-1,f],RsStat_PAC[-2,f]),0)
# y2 = np.concatenate((RsStat_PIC[-1,f],RsStat_PIC[-2,f]),0)
# res = stats.ks_2samp(y1,y2)
# print(res[1])
# print(np.mean(y1))
# print(np.mean(y2))


# for c_ind in c_list:
# for f in np.arange(1,ax_sz+1):    
#     s = stats.ks_2samp(Rsstat[-2,f],Rsstat[-1,f])
#     print(s[1])

# axes[0].scatter(np.ones_like(Rscore[c_ind][4,d_list])*4,Rscore[c_ind][4,d_list])



# axes[1].scatter(scatter_ind,Rscore[-1])
# %% 











# %% good fit list

# good_list = good_list_int
d_list = good_list > 179

d_list3 = good_list <= 179

good_list_sep = good_list[d_list3]

weight_thresh = 2*1e-2


def create_convdata():    
    Convdata = {};
    good_fit_list = {} 
    
    for c_ind in c_list:
        good_fit_list[c_ind] = np.zeros((1,np.size(good_list_sep,0)))
        # if c_ind == -1 or c_ind == -2:
        #     ax_sz = 4
        #     cmap3 = ['tab:orange','tab:purple','tab:red','tab:olive']
        
            
        # elif c_ind == 1 or c_ind == 2:
        #     ax_sz = 4
        #     cmap3 = ['tab:orange','tab:blue']
            
        
        
        # for b_ind in np.arange(ax_sz):
        #     Convdata[c_ind,b_ind] = np.zeros((np.size(good_list_sep),np.size(score,0)))
            
        for n in np.arange(np.size(good_list_sep,0)):
            

            # n = int(n)
            nn = good_list_sep[n]
            nn = int(nn)
            X, Y, L, Rt = import_data_w_Ca(D_ppc,nn,prestim,t_period,window,c_ind)

            Model_coef = Data[nn, c_ind-1]["coef"]
            Model_score = Data[nn, c_ind-1]["score"]
            I = Data[nn,c_ind-1]["intercept"]
            X[X[:,0] == 0,0] = -1
            norm_score = np.mean(Model_score, 1)
            norm_score[norm_score < weight_thresh] = 0
            
            # norm_score = Model_score*Model_score*1e4
            if np.mean(norm_score*norm_score*1e4) > weight_thresh*1e2:
                good_fit_list[c_ind][0,n] = 1
            Model_coef = Model_coef/(np.max(np.abs(Model_coef)) + 0.2) # soft normalization value for model_coef
  
            # conv = Model_coef*norm_score
            CP = {}
            for b_ind in np.arange(1,np.size(Model_coef, 0)):
                CP[b_ind-1] = np.outer(X[:,b_ind-1],Model_coef[b_ind,:])
            
            Convdata[n,c_ind] = {"X" : X, "intercept" : I, "component":CP}
    return Convdata, good_fit_list


Convdata, good_fit_list = create_convdata()

# %% 

R = {}

for c_ind in c_list:
    for i in [0,1]:
        R[c_ind,i] = np.zeros((np.size(good_fit_list[c_ind],1),160))
        
        
    for n in np.arange(np.size(good_fit_list[c_ind],1)):
        X = Convdata[n,c_ind]["X"]
        cp = Convdata[n,c_ind]["component"]
        if good_fit_list[c_ind][0,n] > 0:
            R[c_ind,0][n,:] = np.mean(cp[0][X[:,0]==-1,:],0)
            R[c_ind,1][n,:] = np.mean(cp[0][X[:,0]==1,:],0)
            
            

fig, axes = plt.subplots(1,1, figsize = (10,8))

for c_ind in c_list:
    for i in [0,1]:
        axes.plot(x_axis,np.mean(R[c_ind,i],0))
                       








