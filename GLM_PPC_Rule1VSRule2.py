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
    Yraw = {}
    Yraw = D_ppc[n,0]
    Yraw2 = np.concatenate((np.flip(Yraw[0,0:3000],0),Yraw[0,:],Yraw[0,-3000:-1]),0)
    sliding_w= np.lib.stride_tricks.sliding_window_view(np.arange(np.size(Yraw,1)+6000), 6000)
    Ymed_wind = np.zeros((1,np.size(Yraw,1)))
    for s in np.arange(np.size(Yraw,1)):
        Ymed_wind[0,s] = np.median(Yraw2[sliding_w[s,:]])
        
    Yraw3 = Yraw-Ymed_wind+np.mean(Yraw)
    Y = np.zeros((N_trial,int(t_period/window)))
    for tr in range(N_trial):
        Y[tr,:] = Yraw3[0,D_ppc[n,2][tr,0]-1 - int(prestim/window): D_ppc[n,2][tr,0] + int(t_period/window)-1 - int(prestim/window)]

    # Y = np.zeros((N_trial,int(t_period/window)))
    # for tr in range(N_trial):
    #     Y[tr,:] = D_ppc[n,0][0,D_ppc[n,2][tr,0]-1 - int(prestim/window): D_ppc[n,2][tr,0] + int(t_period/window)-1 - int(prestim/window)]
                
    
                
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
        r_ind = np.arange(200,np.size(X,0))
        
    
        
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
    X2 = X2[r_ind,:]
    L2 = L2[r_ind,:]

    # X = X[:,[1,3,4]] # removing contingency and correct
        
        
    return X2,Y, L2, Rt 


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
        
        X3 = np.column_stack([l,X])
        # X3 = np.column_stack([X10,X])
        X4  = X3
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
        # X3[X3 == 0] = -1
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
            cmap = ['tab:orange','tab:blue','tab:red','tab:olive']
            clabels = ["lick","stim","Reward","history"]
            lstyles = ['solid','solid','solid','dashed']
            
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
        
    
        ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind1,:],0),2),
                 linewidth = 2.0, color = cmap[2],label = '5 kHz',linestyle = lstyles[2])
        ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind2,:],0),2),
                 linewidth = 2.0, color = cmap[2],label = '10 kHz',linestyle = lstyles[3])
        ax1.set_title('Firing rate y')
        ax1.legend(loc = 'upper right')
    
        
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

# %% 

def build_model(n, t_period, prestim, window,k,c_ind,ca):
    for m_ind in [0,1,2,3]:
        X, Y, Yhat, Model_Theta, score, inter = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,m_ind,0)
        Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score, 'Y' : Y,'Yhat' : Yhat}
        mi, bs, coef,beta_weights,mean_score, var_score,score_pool = Model_analysis(n, window, window2, Data,c_ind,ana_period)
        S[0,m_ind] = mean_score[0,mi]
        mean_score[mean_score<weight_thresh] = 0
        DataS[n,c_ind-1,m_ind] = {"mean_score" : mean_score, "var_score" : var_score,"score_pool" : score_pool}
    maxS = np.argmax(S)
    max_score_pool = DataS[n,c_ind-1,maxS]["score_pool"]
    
    it = 0
    while it < 4:
        p = np.zeros((np.size(X,1),np.size(max_score_pool,1)))
        mean_score_pool = np.zeros((np.size(X,1),np.size(max_score_pool,1)))
        if np.any(DataS[n,c_ind-1,np.argmax(S)]["mean_score"] >  DataS[n,c_ind-1,np.argmax(S)]["var_score"]):
            for m_ind in [0,1,2,3]:
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
            it = 4   
    
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
ax_sz = 4
S = np.zeros((1,ax_sz))
ana_period = np.array([0, t_period+prestim])
weight_thresh = 2*1e-2

# change c_ind and n here. 
# good_list3 = {}
# for c_ind in c_list:
#     t = 0 
#     good_list2 = [];
    # good_list3[c_ind] =[];
for n in good_list:
    for c_ind in c_list:
        n = int(n)
        # X, Y, Yhat, Model_Theta, score = glm_per_neuron(n, t_period, prestim, window,k,c_ind)
        # Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score} 
        try:
            
            maxS = build_model(n, t_period, prestim, window, k, c_ind, ca)
            # maxS = Data[n,c_ind-1]["maxS"]
            # try: 
            #     maxS.remove(0)
            # except:
            #     maxS = maxS;

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

# np.save('R1vsR2Data_0530.npy',Data, allow_pickle = True)

Data = np.load('R1vsR2Data_real2.npy',allow_pickle= True).item()


# %% Calculating best_kernel
# good_list_int = np.intersect1d(good_list3[-1],good_list3[-2])
# good_list2 = good_list_int
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
    for n in good_list:
        n = int(n)
        mi, bs, coef,beta_weights,mean_score,score_var,score_pool = Model_analysis(n, window, window2, Data,c_ind,ana_period)
        norm_coef = np.abs(coef)
        # Y_mean = np.mean(Data[n,c_ind-1]["Y"])
        if bs > weight_thresh:
            best_kernel[c_ind][0,k] = int(mi)
            best_kernel[c_ind][1,k] = int(np.argmax(np.abs(coef)))+1
            best_kernel[c_ind][2,k] = norm_coef[0] 
            best_kernel[c_ind][3,k] = norm_coef[1]
            best_kernel[c_ind][4,k] = norm_coef[2]
            if c_ind ==1 or c_ind == 2:  
                best_kernel[c_ind][5,k] = norm_coef[3]

            elif c_ind == -1 or c_ind == -2:
                best_kernel[c_ind][5,k] = norm_coef[3]
        else:
            best_kernel[c_ind][2:b_ind,k] = np.ones((1,b_ind-2))*-1    
        k = k+1
        
    return best_kernel

weight_thresh = 1*1e-2


# Here we define the time period for model analysis. 
# ana_period = np.array([2000, 4000]) # (Stimulus presentation period)
# ana_period = np.array([1500, 2500])
# ana_period = np.array([2500, 4500])
ana_period = np.array([0, 4500])
for c_ind in c_list:
    if c_ind == -1 or c_ind == -2 :
        b_ind = 6
    else:
        b_ind = 5
        
    best_kernel = get_best_kernel(b_ind, window, window2, Data, c_ind, ana_period,good_list)
        
# %% Accumulated task variable encoding piechart 04/17

def pie_accumulated(Data,best_kernel, c_ind):
    d_list = good_list > 179

    d_list3 = good_list <= 179
    
    # good_list_sep = np.int_(good_list[d_list])
    b_list = np.arange(np.size(good_list))
    b_list = b_list[d_list]
    pie_labels = [ "lick","stim","reward","history"]
    cmap = ['tab:orange','tab:blue','tab:red','tab:olive'] 
    cat_concat = [];
    for n in b_list:
        if best_kernel[c_ind][1,n] > 0:
            maxSn = Data[int(good_list[n]),c_ind-1]["maxS"]
            if type(maxSn) is np.int64:
                maxSn = [maxSn]
            
            cat_concat = np.concatenate((cat_concat,maxSn))
            # except:
            #     cat_concat = np.concatenate((cat_concat,[Data[int(good_list[n]),c_ind-1]["maxS"]]))
                
    plt.pie(np.bincount(cat_concat.astype(int)),labels = pie_labels, colors = cmap)
    print(np.bincount(cat_concat.astype(int)))

    plt.show() 
                
pie_accumulated(Data,best_kernel, -2)



# %% bar chart regarding pie chart
cmap = ['tab:orange','tab:blue','tab:red','tab:olive'] 

# b1 = [31, 67, 16, 57]
# b2 = [25, 49, 51, 59]
# b3 = [9, 43, 14, 42]




#  ppc_AC
b1 = [46, 107,  68, 132]
b2 = [52,  70,  74, 106]
b3 = [9, 37, 37, 77]

fig, axes = plt.subplots(1,1,figsize = (10,8))
axes.bar(np.arange(4)*2,b1, color = cmap, alpha = 0.5)
axes.bar(np.arange(4)*2+1,b2, color = cmap, alpha = 1)
axes.bar(np.arange(4)*2,b3, color = 'tab:gray', alpha = 0.7)
axes.bar(np.arange(4)*2+1,b3, color = 'tab:gray', alpha = 0.7)
axes.set_xlim([1.5, 7.55])


# %% task variable evolution per unit

def TV_encoding(Data,best_kernel,d_list):
    b_list = np.arange(np.size(good_list))
    b_list = b_list[d_list]
    ax_sz = 4
    TV = {}
    TV[0] = np.zeros((1,3))
    TV[1] = np.zeros((1,3))
    TV[2] = np.zeros((1,3))
    TV[3] = np.zeros((1,3))
    for n in b_list:
        
        maxSn1 = Data[int(good_list[n]),-2]["maxS"]
        maxSn2 = Data[int(good_list[n]),-3]["maxS"]
        if type(maxSn1) is np.int64:
            maxSn1 = [maxSn1]
        if type(maxSn2) is np.int64:
            maxSn2 = [maxSn2]
        
        
        if best_kernel[-1][1,n] > 0 and best_kernel[-2][1,n] > 0 :

            for f in np.arange(ax_sz):
                if f in maxSn1 and f in maxSn2:
                    TV[f][0,2] += 1
                elif f in maxSn1 and f not in maxSn2:
                    TV[f][0,0] += 1
                elif f not in maxSn1 and f in maxSn2:
                    TV[f][0,1] += 1
    
    return TV
                    
d_list = good_list > 179

d_list3 = good_list <= 179                
        
tv = TV_encoding(Data,best_kernel,d_list3)

# %% pool neurons encoding each variable.

def make_TV_list(Data,best_kernel,d_list):
    b_list = np.arange(np.size(good_list))
    b_list = b_list[d_list]
    ax_sz = 4
    tvlist = {}
    for f in np.arange(ax_sz):
        tvlist[f] = {}
        tvlist[f][0] = []
        tvlist[f][1] = []
        tvlist[f][2] = []
            

    for n in b_list:
        
        maxSn1 = Data[int(good_list[n]),-2]["maxS"]
        maxSn2 = Data[int(good_list[n]),-3]["maxS"]
        if type(maxSn1) is np.int64:
            maxSn1 = [maxSn1]
        if type(maxSn2) is np.int64:
            maxSn2 = [maxSn2]
            
        for f in np.arange(ax_sz):
            if f in maxSn1 and f in maxSn2:
                if best_kernel[-1][1,n] > 0 and best_kernel[-2][1,n] > 0 :
                    tvlist[f][0].append(n)
            elif f in maxSn1 and f not in maxSn2:
                if best_kernel[-1][1,n] > 0:
                    tvlist[f][1].append(n)
            elif f not in maxSn1 and f in maxSn2:
                if best_kernel[-2][1,n] > 0:
                    tvlist[f][2].append(n)
                    
    return tvlist

tvlist1 = make_TV_list(Data,best_kernel,d_list3) # Make tvlist for PPC_IC or PPC_AC
tvlist2 = make_TV_list(Data,best_kernel,d_list)
                    
np.save("tvlist_PIC.npy",tvlist1)    
np.save("tvlist_PAC.npy",tvlist2)    


# %% Normalized population average of task variable weights
# good_list = good_list_int
d_list = good_list > 179

d_list3 = good_list <= 179

good_list_sep = good_list[:]

weight_thresh = 4*1e-2

score = np.arange(160)

Convdata = {};

for c_ind in c_list:
    if c_ind == -1 or c_ind == -2:
        ax_sz = 4
        cmap3 = ['tab:orange','tab:blue','tab:red','tab:olive']
    
        
    elif c_ind == 1 or c_ind == 2:
        ax_sz = 4
        cmap3 = ['tab:orange','tab:blue']
        
    
    
    for b_ind in np.arange(ax_sz):
        Convdata[c_ind,b_ind] = np.zeros((np.size(good_list_sep),np.size(score,0)))
        
    for n in np.arange(np.size(good_list_sep,0)):
        # n = int(n)
        nn = good_list_sep[n]
        Model_coef = Data[nn, c_ind-1]["coef"]
        Model_score = Data[nn, c_ind-1]["score"]
        # if c_ind == -2: # for rule 2, switch signs for stim, this makes it into contingency
        #     Model_coef[1,:] = -Model_coef[1,:]
    
        # Model_coef = np.abs(Model_coef)/(np.max(np.abs(Model_coef)) + 0.2) # soft normalization value for model_coef
        Model_coef = Model_coef/(np.max(np.abs(Model_coef)) + 0.2) # soft normalization value for model_coef
        
        norm_score = np.mean(Model_score, 1)
        norm_score[norm_score < weight_thresh] = 0
        norm_score[norm_score > 0] = 1
        # if np.max(norm_score)>0:
        #     norm_score = norm_score/np.max(norm_score)
        #     # norm_score = 1
        # else:
        #     norm_score = 0    
        conv = Model_coef*norm_score
        # if np.mean(norm_score*norm_score*1e4) > weight_thresh*1e2:
        #     conv = Model_coef
        # else:
        #     conv = Model_coef*0
        for b_ind in np.arange(np.size(Model_coef, 0)):
            Convdata[c_ind,b_ind][n, :] = conv[b_ind, :]
    
    
    x_axis = np.arange(1, prestim+t_period, window)
    fig, axes = plt.subplots(1,1,figsize = (10,8))
    
    for f in range(1,ax_sz):
            # plotting only units with non-zero weights
            
            C = Convdata[c_ind,f]
            C = C[np.max(np.abs(C),1) > 0,:]
            error = np.std(C,0)/np.sqrt(np.size(C,0))
            y = ndimage.gaussian_filter(np.mean(C,0),2)
            # error = np.std(Convdata[c_ind,f],0)/np.sqrt(np.size(good_list_sep))
            # y = ndimage.gaussian_filter(np.mean(Convdata[c_ind,f],0),1)
            axes.plot(x_axis*1e-3-prestim*1e-3,y,c = cmap3[f])
            axes.fill_between(x_axis*1e-3-prestim*1e-3,y-error,y+error,facecolor = cmap3[f],alpha = 0.3)
            axes.set_ylim([-0.2,0.4])
    
    
    e_lines = np.array([0, 500, 500+1000, 2500+1000])
    e_lines = e_lines+500

# np.save('D:\Python\ConvR1vsR2_cont.npy',Convdata)



# %% separate neurons with 5khz coding and 10khz coding

c_ind = -2
d_list5khz = np.zeros((1,294))
d_list10khz = np.zeros((1,294))

p = np.arange(100,160) # stim period
# p = np.arange(60,100)

for n in np.arange(95,294):
    if np.mean(Convdata[c_ind,1][n,p]) > np.std(Convdata[c_ind,1][n,p]):
        d_list5khz[0,n] = 1
    if np.mean(Convdata[c_ind,1][n,p]) < -np.std(Convdata[c_ind,1][n,p]):
        d_list10khz[0,n] = 1

d_list5khz = (d_list5khz >0)  
d_list10khz = (d_list10khz >0)        
      

fig, axes = plt.subplots(2,2,figsize = (12,10))

axes[0,0].plot(x_axis*1e-3-prestim*1e-3,np.mean(Convdata[c_ind,1][d_list5khz[0],:],0),c = "tab:blue")
axes[1,0].plot(x_axis*1e-3-prestim*1e-3,np.mean(-Convdata[c_ind,1][d_list10khz[0],:],0).T,c = "tab:blue")
axes[0,0].plot(x_axis*1e-3-prestim*1e-3,np.mean(Convdata[c_ind,2][d_list5khz[0],:],0).T,c = "tab:red")
axes[1,0].plot(x_axis*1e-3-prestim*1e-3,np.mean(Convdata[c_ind,2][d_list10khz[0],:],0).T,c = "tab:red") 
# axes[0,0].plot(x_axis*1e-3-prestim*1e-3,np.mean(Convdata[c_ind,3][d_list5khz[0],:],0).T,c = "tab:olive")
# axes[1,0].plot(x_axis*1e-3-prestim*1e-3,np.mean(-Convdata[c_ind,3][d_list10khz[0],:],0).T,c = "tab:olive")      

# %%

weight = {}
# p = np.arange(0,25)
p = np.arange(25,50)


    
for f in np.arange(ax_sz):  
    weight[-1,f]= np.zeros((1,294))
    weight[-2,f] = np.zeros((1,294))    
    for c_ind in c_list:
        for n in np.arange(95,294):
            weight[c_ind,f][0,n] = np.mean(Convdata[c_ind,f][n,p])
            
fig, axes = plt.subplots(1,1,figsize = (10,8))
axes.scatter(weight[-1,1],-weight[-2,1])
axes.set_xlim([-0.8,0.8])
axes.set_ylim([-0.8,0.8])


# %%    

f = 1
list2 = (weight[-1,f] < -0.1) #* (weight[-2,f] == 0)
list3 = (weight[-2,f] > 0.1)# * (weight[-1,f] == 0)
print(np.sum(list2))
print(np.sum(list3))
# list4 = (weight[-1,f] > 0)*(-weight[-2,f] > 0)
# result = stats.linregress(weight[-1,f][list4],-weight[-2,f][list4])

# print(result.rvalue)

# 
fig, axes = plt.subplots(1,1,figsize = (10,8))

for f in [1]:
    
    
    C = np.concatenate((Convdata[-1,f][list2[0,:],:], -Convdata[-2,f][list3[0,:],:]))
    C = C[np.max(np.abs(C),1)>0.1,:]
    y1 = np.mean(C,0)
    s1 = np.std(C,0)/np.sqrt(np.size(C,0))
    
    # y1 = np.mean(Convdata[-1,f][list2[0,:],:],0)
    # s1 = np.std(Convdata[-1,f][list2[0,:],:],0)/np.sqrt(np.sum(list2))
    y2 = np.mean(Convdata[-2,f][list3[0,:],:],0)
    s2 = np.std(Convdata[-2,f][list3[0,:],:],0)/np.sqrt(np.sum(list3))
    
    # take history units
    
    
    
    # p1 = np.arange(90,110)
    t1 = Convdata[-1,f][list2[0,:],:]
    
    t2 = Convdata[-2,f][list3[0,:],:]
    # stats.ks_2samp(np.mean(t1,1),np.mean(t2,1))
    
    y1 = ndimage.gaussian_filter(y1,2)
    y2 = ndimage.gaussian_filter(y2,2)
    
    cmap = cmap3 = ['tab:orange','tab:blue','tab:red','tab:olive']
    
    axes.plot(x_axis*1e-3-prestim*1e-3,y1,c = cmap[f],linestyle = 'dashed')
    axes.fill_between(x_axis*1e-3-prestim*1e-3,y1-s1,y1+s1,facecolor = cmap[f],alpha = 0.3)
    
    # axes.plot(x_axis*1e-3-prestim*1e-3,y2,c = cmap[f],linestyle = 'dashed')
    # axes.fill_between(x_axis*1e-3-prestim*1e-3,y2-s2,y2+s2,facecolor = cmap[f],alpha = 0.3)
    
    axes.legend(["R1","","R2"])
    # axes.set_ylim([-0.12,0.05])
    plt.savefig("PPC_IC_hist3.svg")


# axes.plot(x_axis*1e-3-prestim*1e-3,np.mean(Convdata[c_ind,1][d_list10khz[0],:],0).T,c = "tab:blue")
# axes[0,1].plot(x_axis*1e-3-prestim*1e-3,np.mean(Convdata[c_ind,2][d_list5khz[0],:],0).T,c = "tab:red")
# axes[1,1].plot(x_axis*1e-3-prestim*1e-3,np.mean(Convdata[c_ind,2][d_list10khz[0],:],0).T,c = "tab:red")      

print(np.sum(list2+list3))
# %% Use model weight to plot units
c_ind = -1
for n in good_list[list2[0]]:
    n = int(n)
        # X, Y, Yhat, Model_Theta, score = glm_per_neuron(n, t_period, prestim, window,k,c_ind)
        # Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score} 
    maxS = Data[n,c_ind-1]["maxS"]
            # maxS = [0,1,2,3]  
    X, Y, Yhat, Model_Theta, score, intercept = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca,maxS,1)




# %% PCA 
# do PCA multiple times with 90% shuffling


n_cv = 100
max_k = 30
Overlap = {};
Overlap[c_list[0]] = np.zeros((ax_sz,ax_sz,n_cv));
Overlap[c_list[1]] = np.zeros((ax_sz,ax_sz,n_cv));
Overlap_across = np.zeros((ax_sz,ax_sz,n_cv));
O_mean = np.zeros((ax_sz,ax_sz));
O_std = np.zeros((ax_sz,ax_sz));

tvlistnew = {}
for k in np.arange(n_cv):
    d_list3 = good_list <= 179
    d_list = good_list > 179
 
    for s in np.arange(np.size(good_list)):
        if d_list[s] == True:
            shuffle = np.random.choice(2,1, p = [0.9,0.1])
            if shuffle == 1:
                d_list[s] = False
        
        if d_list3[s] == True:
            shuffle = np.random.choice(2,1, p = [0.9,0.1])
            if shuffle == 1:
                d_list3[s] = False

    d_list = good_list > 179

    d_list3 = good_list <= 179

    # for f in np.arange(1,ax_sz):    
    #     tvlistnew[f] = np.random.choice(tvlist1[f][0],int(np.floor(np.size(tvlist1[f][0])*0.9)),replace=False)

    pca = {};
    
    
    for c_ind in c_list:
        # fig, axs = plt.subplots(ax_sz,6,figsize = (20,20))
        for f in np.arange(1,ax_sz):
            # pca[f] = SparsePCA(n_components=10,alpha = 0.01)  
            pca[c_ind,f] = PCA(n_components=max_k) 
            # test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(Convdata[c_ind,f][tvlistnew[f],:].T,[1,0]))
            # if f == 1 & c_ind == -2:
            #     test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(-Convdata[c_ind,f][d_list3,:].T,[1,0]))
            # else:
            test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(Convdata[c_ind,f][d_list3,:].T,[1,0]))
            # test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(Convdata[c_ind,f][d_list3,:].T,[1,0]))
            # test = test.T
            # for t in range(5):
            #     axs[f,t].plot(test[t,:],c = cmap3[f])
            # axs[f,5].plot(np.cumsum(pca[c_ind,f].explained_variance_ratio_))
            # plt.savefig("test.svg", format = 'svg')
    
    for f in np.arange(1,ax_sz):
        for f2 in np.arange(1,ax_sz):
            for c_ind in c_list:
                S_value = np.zeros((1,max_k))
                
                for d in np.arange(0,max_k):
                    S_value[0,d] = np.abs(np.dot(pca[c_ind,f].components_[d,:], pca[c_ind,f2].components_[d,:].T))
                    S_value[0,d] = S_value[0,d]/(np.linalg.norm(pca[c_ind,f].components_[d,:])*np.linalg.norm(pca[c_ind,f2].components_[d,:]))
                        
                Overlap[c_ind][f,f2,k] = np.max(S_value)
            
            for d in np.arange(0,max_k):
                S_value[0,d] = np.abs(np.dot(pca[-1,f].components_[d,:], pca[-2,f2].components_[d,:].T))
                S_value[0,d] = S_value[0,d]/(np.linalg.norm(pca[-1,f].components_[d,:])*np.linalg.norm(pca[-2,f2].components_[d,:]))
            
            Overlap_across[f,f2,k] = np.max(S_value)
                
                # Overlap[c_ind][f,f2,k] = np.max(np.abs(np.dot(pca[c_ind,f].components_, pca[c_ind,f2].components_.T)*np.identity(20)))
                # Overlap_across[f,f2,k] = np.max(np.abs(np.dot(pca[c_list[0],f].components_, pca[c_list[1],f2].components_.T)*np.identity(20)))
        
    

                   

    

    #   Subspace overlap analysis







for f in np.arange(1,ax_sz):
    for f2 in np.arange(1,ax_sz):        
        O_mean[f,f2] = np.mean(Overlap_across[f,f2,:])
        O_std[f,f2] = np.std(Overlap_across[f,f2,:])



# x1 = [.8,1.8,2.8]
# y1 = [O_mean[0,0],O_mean[1,1],O_mean[2,2]]
# e1 = [O_std[0,0],O_std[1,1],O_std[2,2]]


# %% dendrogram

from scipy.cluster.hierarchy import dendrogram, linkage


O_mean2 = np.concatenate((np.concatenate((np.mean(Overlap[-1],2),O_mean.T),axis = 1)
            ,np.concatenate((O_mean,np.mean(Overlap[-2],2)),axis = 1)),axis = 0)


Z = linkage(O_mean2,'complete')

fig, ax = plt.subplots(1,1,figsize =  (10,8))
dn1 = dendrogram(Z,labels = ['R1_Lick','R1_Stim','R1_Rew','R1_Hist','R2_Lick','R2_Stim','R2_Rew','R2_Hist'])



# %% Temporal dynamics of variance encoding through PCs
# 1. Run PCA separately. 

      
def temporal_var_exp(Convdata,pca,d_list,tlist):
    l_styles = ['solid','dotted']
    array_length = np.size(Convdata[-1,0],1)
    
    xtime = np.arange(array_length)*50*1e-3-prestim*1e-3
    
    n_pc = 10
    n_pc1 = 0
    n_cv = 20
    V_cap = {}
    V_cap1 = {}
    for c_ind in c_list:
        V_cap[c_ind]  =np.zeros((ax_sz,array_length,n_cv))
        V_cap1[c_ind]  =np.zeros((ax_sz,array_length,n_cv))
    V_cap1_base = np.zeros((ax_sz,n_cv))
    # V_cap2_base = np.zeros((ax_sz,n_cv))
    
    R = {}
    for f  in np.arange(1,ax_sz): 

        
        for cv in np.arange(n_cv):
            for c_ind in c_list:
                # if not tlist:
                R[c_ind] = ndimage.gaussian_filter(Convdata[c_ind,f][:,:].T,[1,0])
                # else:
                    # R[c_ind] = ndimage.gaussian_filter(Convdata[c_ind,f][tlist[f][0],:].T,[1,0])

                # R0 = ndimage.gaussian_filter(Convdata[c_ind,f][:,:].T,[1,0])
            # create baseline explained variance with shuffled data
            # r_shuffle = np.arange(len(good_list))
            # np.random.shuffle(r_shuffle)
            # R2 = R0[:,r_shuffle]
            # if not tlist:
            #     R2 = R2[:,d_list]
            # else:
            #     R2 = R2[:,tlist[f][0]]    
            # V_cap1_base[f,cv] = 1-np.linalg.norm(R2 - np.dot(np.dot(R2,pca[c_ind,f].components_[n_pc1:n_pc,:].T),
            #                                                         pca[c_ind,f].components_[n_pc1:n_pc,:]))/np.linalg.norm(R2)
            
            # Shuffle d_list by removing 10% 
            # if not tlist:
            #     s_list = np.random.choice(np.arange(np.sum(d_list)),int(np.floor(np.sum(d_list)*0.90)),replace=False)
            # # s_list = np.arange(np.sum(d_list))
            # else:
            #     s_list = np.random.choice(np.arange(np.size(tlist[f][0])),int(np.size(tlist[f][0])*0.9),replace = False)
            ss_list = {}
            s_list =  [item-95 for item in tlist[f][0]]
            ss_list[1] = [item-95 for item in tlist[f][1]]
            ss_list[2] = [item-95 for item in tlist[f][2]]

            for c_ind in c_list:
                
                # R[c_ind] = R[c_ind][:,s_list]                
                for t in np.arange(array_length):                        
                    V_cap[c_ind][f,t,cv] = 1-np.linalg.norm(R[c_ind][t,tlist[f][0]] - np.dot(np.dot(R[c_ind][t,tlist[f][0]],
                                                                          pca[c_ind,f].components_[n_pc1:n_pc,s_list].T),
                                                                            pca[c_ind,f].components_[n_pc1:n_pc,s_list]))/np.linalg.norm(R[c_ind][t,tlist[f][0]])
                    
                    
                    V_cap1[c_ind][f,t,cv] = 1 - np.linalg.norm(R[c_ind][t,d_list] - np.dot(np.dot(R[c_ind][t,d_list],
                                                                          pca[c_ind,f].components_[n_pc1:n_pc,:].T),
                                                                            pca[c_ind,f].components_[n_pc1:n_pc,:]))/np.linalg.norm(R[c_ind][t,d_list])
                    
                    # V_cap1[c_ind][f,t,cv] = 1-np.linalg.norm(R[c_ind][t,tlist[f][-c_ind]] - np.dot(np.dot(R[c_ind][t,tlist[f][-c_ind]],
                    #                                                       pca[c_ind,f].components_[n_pc1:n_pc,ss_list[-c_ind]].T),
                    #                                                         pca[c_ind,f].components_[n_pc1:n_pc,ss_list[-c_ind]]))/np.linalg.norm(R[c_ind][t,tlist[f][-c_ind]])
                           
        
        
    for c_ind in c_list:                
        fig, axes = plt.subplots(ax_sz,1,figsize = (5,10))
        for f  in np.arange(1,ax_sz): 
        # for c_ind in c_list:
            axes[f].plot(xtime, np.mean(V_cap[c_ind][f,:,:],1),c = cmap3[f],linestyle = 'solid')#l_styles[np.abs(c_ind)-1])
            axes[f].fill_between(xtime,np.mean(V_cap[c_ind][f,:,:],1)- np.std(V_cap[c_ind][f,:,:],1),np.mean(V_cap[c_ind][f,:,:],1)+ np.std(V_cap[c_ind][f,:,:],1),facecolor = cmap3[f],alpha = 0.2)
            
            axes[f].plot(xtime, np.mean(V_cap1[c_ind][f,:,:],1),c = cmap3[f],linestyle = 'dotted')#l_styles[np.abs(c_ind)-1])
            axes[f].fill_between(xtime,np.mean(V_cap1[c_ind][f,:,:],1)- np.std(V_cap1[c_ind][f,:,:],1),np.mean(V_cap1[c_ind][f,:,:],1)+ np.std(V_cap1[c_ind][f,:,:],1),facecolor = cmap3[f],alpha = 0.2)
            
            y =np.mean(V_cap1_base[f,:])
                # error = np.std(V_cap1_base[f,:])
            axes[f].hlines(y, 
                      xmin = min(xtime), 
                      xmax = max(xtime),
                      linestyles = 'dashed',
                      colors = 'black', 
                      linewidth = 2.0)
            # axes[f].set_ylim =
            
def temporal_var_exp2(Convdata,pca,d_list):
    l_styles = ['solid','dotted']
    array_length = np.size(Convdata[-1,0],1)
    
    xtime = np.arange(array_length)*50*1e-3-prestim*1e-3
    
    n_pc = 3
    n_pc1 = 0
    n_cv = 20
    V_cap = {}
    V_cap1 = {}
    for c_ind in c_list:
        V_cap[c_ind]  =np.zeros((ax_sz,array_length,n_cv))
        V_cap1[c_ind]  =np.zeros((ax_sz,array_length,n_cv))
    V_cap1_base = np.zeros((ax_sz,n_cv))
    # V_cap2_base = np.zeros((ax_sz,n_cv))
    
    R = {}
    for f  in np.arange(1,ax_sz): 

        
        for cv in np.arange(n_cv):
            for c_ind in c_list:
                # if not tlist:
                R[c_ind] = ndimage.gaussian_filter(Convdata[c_ind,f][:,:].T,[1,0])
                # else:
                    # R[c_ind] = ndimage.gaussian_filter(Convdata[c_ind,f][tlist[f][0],:].T,[1,0])

                # R0 = ndimage.gaussian_filter(Convdata[c_ind,f][:,:].T,[1,0])
            # create baseline explained variance with shuffled data
            # r_shuffle = np.arange(len(good_list))
            # np.random.shuffle(r_shuffle)
            # R2 = R0[:,r_shuffle]
            # if not tlist:
            #     R2 = R2[:,d_list]
            # else:
            #     R2 = R2[:,tlist[f][0]]    
            # V_cap1_base[f,cv] = 1-np.linalg.norm(R2 - np.dot(np.dot(R2,pca[c_ind,f].components_[n_pc1:n_pc,:].T),
            #                                                         pca[c_ind,f].components_[n_pc1:n_pc,:]))/np.linalg.norm(R2)
            
            # Shuffle d_list by removing 10% 
            # if not tlist:
            #     s_list = np.random.choice(np.arange(np.sum(d_list)),int(np.floor(np.sum(d_list)*0.90)),replace=False)
            # # s_list = np.arange(np.sum(d_list))
            # else:
            #     s_list = np.random.choice(np.arange(np.size(tlist[f][0])),int(np.size(tlist[f][0])*0.9),replace = False)
                
                # R[c_ind] = R[c_ind][:,s_list]   
            for c_ind in c_list:
                for t in np.arange(array_length):                        
                    V_cap[c_ind][f,t,cv] = 1-np.linalg.norm(R[c_ind][t,d_list] - np.dot(np.dot(R[c_ind][t,d_list],
                                                                          pca[c_ind,f].components_[n_pc1:n_pc,:].T),
                                                                            pca[c_ind,f].components_[n_pc1:n_pc,:]))/np.linalg.norm(R[c_ind][t,d_list])
                    
            for t in np.arange(array_length):                        
                    V_cap1[-1][f,t,cv] = 1 - np.linalg.norm(R[-1][t,d_list] - np.dot(np.dot(R[-1][t,d_list],
                                                                          pca[-2,f].components_[n_pc1:n_pc,:].T),
                                                                            pca[-2,f].components_[n_pc1:n_pc,:]))/np.linalg.norm(R[-1][t,d_list])
                    
                    V_cap1[-2][f,t,cv] = 1 - np.linalg.norm(R[-2][t,d_list] - np.dot(np.dot(R[-2][t,d_list],
                                                                          pca[-1,f].components_[n_pc1:n_pc,:].T),
                                                                            pca[-1,f].components_[n_pc1:n_pc,:]))/np.linalg.norm(R[-2][t,d_list])                   
                    # V_cap1[c_ind][f,t,cv] = 1-np.linalg.norm(R[c_ind][t,tlist[f][-c_ind]] - np.dot(np.dot(R[c_ind][t,tlist[f][-c_ind]],
                    #                                                       pca[c_ind,f].components_[n_pc1:n_pc,ss_list[-c_ind]].T),
                    #                                                         pca[c_ind,f].components_[n_pc1:n_pc,ss_list[-c_ind]]))/np.linalg.norm(R[c_ind][t,tlist[f][-c_ind]])
                           
        
        
    for c_ind in c_list:                
        fig, axes = plt.subplots(ax_sz,1,figsize = (10,10))
        for f  in np.arange(1,ax_sz): 
        # for c_ind in c_list:
            axes[f].plot(xtime, np.mean(V_cap[c_ind][f,:,:],1),c = cmap3[f],linestyle = 'solid')#l_styles[np.abs(c_ind)-1])
            axes[f].fill_between(xtime,np.mean(V_cap[c_ind][f,:,:],1)- np.std(V_cap[c_ind][f,:,:],1),np.mean(V_cap[c_ind][f,:,:],1)+ np.std(V_cap[c_ind][f,:,:],1),facecolor = cmap3[f],alpha = 0.2)
            
            axes[f].plot(xtime, np.mean(V_cap1[c_ind][f,:,:],1),c = cmap3[f],linestyle = 'dotted')#l_styles[np.abs(c_ind)-1])
            axes[f].fill_between(xtime,np.mean(V_cap1[c_ind][f,:,:],1)- np.std(V_cap1[c_ind][f,:,:],1),np.mean(V_cap1[c_ind][f,:,:],1)+ np.std(V_cap1[c_ind][f,:,:],1),facecolor = cmap3[f],alpha = 0.2)
            
            y =np.mean(V_cap1_base[f,:])
                # error = np.std(V_cap1_base[f,:])
            axes[f].hlines(y, 
                      xmin = min(xtime), 
                      xmax = max(xtime),
                      linestyles = 'dashed',
                      colors = 'black', 
                      linewidth = 2.0)

                           
            
pca = {};
    
max_k = 12    
for c_ind in c_list:
    # fig, axs = plt.subplots(ax_sz,6,figsize = (20,20))
    for f in np.arange(1,ax_sz):
        # pca[f] = SparsePCA(n_components=10,alpha = 0.01)  
        pca[c_ind,f] = PCA(n_components=max_k)
        if f == 1 & c_ind == -2:
            test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(Convdata[c_ind,f][d_list,:].T,[1,0]))
        else:
            test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(Convdata[c_ind,f][d_list,:].T,[1,0]))
        # test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(Convdata[c_ind,f][tvlist2[f][0],:].T,[1,0]))
        # test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(Convdata[c_ind,f][d_list3,:].T,[1,0]))


temporal_var_exp(Convdata,pca,d_list,tvlist1)      
        
# temporal_var_exp2(Convdata,pca,d_list)      

# array_length = np.size(Convdata[-1,0],1)
# V = np.zeros((1,array_length))
# R = ndimage.gaussian_filter(Convdata[c_ind,0][d_list,:].T,[1,0])
# for t in np.arange(array_length):  
#     V[0,t] = np.linalg.norm(R[t,:])

# fig, ax = plt.subplots(1,1,figsize =  (10,8))
    
# ax.plot(np.arange(array_length),V[0,:])

# %% Simply plot data


tlist = tvlist1
lim = {}
lim[1]=[0, 0.4]
lim[2]=[0,0.7]
lim[3]=[0,0.25]

fig, axes = plt.subplots(4,2,figsize = (20,20))
for c_ind in c_list:    
    for f in np.arange(1,ax_sz):
        error = np.std(Convdata[c_ind,f][tlist[f][0],:],0)/np.sqrt(np.size(tlist[f][0]))
        y = ndimage.gaussian_filter(np.mean(Convdata[c_ind,f][tlist[f][0],:],0),1)
        
        error2 = np.std(Convdata[c_ind,f][tlist[f][-c_ind],:],0)/np.sqrt(np.size(tlist[f][-c_ind]))
        y2 = ndimage.gaussian_filter(np.mean(Convdata[c_ind,f][tlist[f][-c_ind],:],0),1)
        
        axes[f,-c_ind -1].plot(x_axis*1e-3-prestim*1e-3,y,c = cmap3[f],linestyle = 'solid')
        axes[f,-c_ind -1].fill_between(x_axis*1e-3-prestim*1e-3,y-error,y+error,facecolor = cmap3[f],alpha = 0.3)
        
        axes[f,-c_ind -1].plot(x_axis*1e-3-prestim*1e-3,y2,c = cmap3[f],linestyle = 'dotted')
        axes[f,-c_ind -1].fill_between(x_axis*1e-3-prestim*1e-3,y2-error2,y2+error2,facecolor = cmap3[f],alpha = 0.3)
        
        axes[f,-c_ind -1].set_ylim(lim[f])
    



# %% draw trajectories


from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

    
def draw_traj(traj,f,v,trmax,sc):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    # styles = ['solid', 'dotted', 'solid','dotted']
    # cmap_names = ['autumn','autumn','winter','winter']
    styles = ['solid','solid','dotted']
    cmap_names = ['autumn','winter','winter']
    for tr in np.arange(trmax):
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
        lc = Line3DCollection(segments, cmap=cmap_names[tr], norm=norm,linestyle = styles[tr])
        lc.set_array(time)
        lc.set_linewidth(2)
        ax.add_collection3d(lc)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        if tr == sc:
            ax.auto_scale_xyz(x,y,z)
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

# %%  Run PCA on model weights, R1 and R2 separately 

c_list = [-1, -2]
cmap3 = ['tab:orange','tab:blue','tab:red','tab:olive']

pca = {}
ax_sz = 4;
max_k = 20;

for c_ind in c_list:
    fig, axs = plt.subplots(ax_sz,6,figsize = (20,20))
    for f in np.arange(1,ax_sz):
        # pca[c_ind,f] = SparsePCA(n_components=10,alpha = 0.01)  
        pca[c_ind,f] = PCA(n_components=max_k)
        # if f == 1 & c_ind == -2:
        #     test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(-Convdata[c_ind,f][d_list,:].T,[1,0]))
        # else:
        #     test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(Convdata[c_ind,f][d_list,:].T,[1,0]))
        # R = Convdata[-1,f][tvlist1[f][0],:] + Convdata[-2,f][tvlist1[f][0],:] 
        if f == 1:
            R = Convdata[-1,f][d_list3,:] - Convdata[-2,f][d_list3,:] 
        else:
            R = Convdata[-1,f][d_list3,:] + Convdata[-2,f][d_list3,:] 

        R = R/2
        test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(R.T,[1,0]))        

        # test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(Convdata[c_ind,f][tvlist1[f][0],:].T,[1,0]))
        test = test.T
        for t in range(5):
            axs[f,t].plot(test[t,:],c = cmap3[f])
        axs[f,5].plot(np.cumsum(pca[c_ind,f].explained_variance_ratio_))
   
    
   
    
# %%   
# pca = np.load('D:\Python\pca_common_all_PIC.npy',allow_pickle= True).item()
traj = {};

# tvlist = tvlist1

for f in np.arange(1,4):
    R1 = ndimage.gaussian_filter(Convdata[-1,f][d_list3,:].T,[2,0])
    if f == 1:
        R2 = ndimage.gaussian_filter(-Convdata[-2,f][d_list3,:].T,[2,0])
    else:
        R2 = ndimage.gaussian_filter(Convdata[-2,f][d_list3,:].T,[2,0])


    # R1 = ndimage.gaussian_filter(D[0,5][d_list3,:].T,[2,0]) 
    # R2 = ndimage.gaussian_filter(D[1,5][d_list3,:].T,[2,0]) 
    traj[f] = {}
    traj[f][0] = np.dot(R1,pca[-1,f].components_.T)  
    traj[f][1] = np.dot(R2,pca[-1,f].components_.T)

# for f in  np.arange(1,5):
#     if f > 1:                 
#         R1 = ndimage.gaussian_filter(Convdata[-1,f-1][tvlist[f-1][0],:].T,[5,0])
#         R2 = ndimage.gaussian_filter(Convdata[-2,f-1][tvlist[f-1][0],:].T,[5,0])
#     else:
#         R1 = ndimage.gaussian_filter(Convdata[-1,f][tvlist[f][0],:].T,[5,0])
#         R2 = ndimage.gaussian_filter(Convdata[-2,f][tvlist[f][0],:].T,[5,0])           
#     traj[f] = {}
#     traj[f][0] = np.dot(R1,pca[f].components_.T)  
#     traj[f][1] = np.dot(R2,pca[f].components_.T)

for f in  np.arange(1,4):
    draw_traj(traj,f,0,2,0)


# %%
D = np.load('D:\Python\Ca_trace.npy',allow_pickle= True).item()
R1 = ndimage.gaussian_filter(D[1,5][d_list,:].T,[2,0]) 
R2 = ndimage.gaussian_filter(D[1,6][d_list,:].T,[2,0]) 
traj[1] = {}
traj[1][0] = np.dot(R1,pca[-1,f].components_.T)  
traj[1][1] = np.dot(R2,pca[-1,f].components_.T)

draw_traj(traj,1,0,2,0)
# %% Calculate trajectory distance

edist = {}

fig, axs = plt.subplots(1,1,figsize = (8,8))
xtime = np.arange(160)*50*1e-3-prestim*1e-3
cmap4 = ["tab:orange","tab:blue","tab:red","tab:olive"]

rms = {}
for f in np.arange(1,4):
    # for t in [0,1,2]:
    #     nmax = np.max(np.abs(np.concatenate(((traj[f][0][:,t],traj[f][1][:,t])))))
    #     traj[f][0][:,t] = traj[f][0][:,t]/nmax
    #     traj[f][1][:,t] = traj[f][1][:,t]/nmax
    
    
    R = np.concatenate((traj[f][0][:,0:3],traj[f][1][:,0:3]),0)
    R= R-np.mean(R,0)
    R1 = np.linalg.norm(R, axis = 1)
    rms[f] = np.sqrt((np.linalg.norm(R1)*np.linalg.norm(R1))/len(R1))
    edist[f] = np.linalg.norm(traj[f][0][:,0:3]-traj[f][1][:,0:3], axis = 1 )# /rms[f]
    axs.plot(xtime,edist[f], c = cmap4[f])
    axs.set_ylim(-0.1,3.1)



    
# %%
# fig, ax = plt.subplots()
# ax.errorbar(x, y, e, linestyle='None', marker='^')
# ax.errorbar(x1, y1, e1, linestyle='None', marker='^')
# ax.set_ylim([0,0.6])

# baseline = np.zeros((18,1))
# k = 0
# for f in np.arange(ax_sz):
#     for f2 in np.arange(ax_sz):
#         for c_ind in c_list:            
#             baseline[k] =  np.max(np.abs(np.dot(pca[c_ind,f].components_[:,arr], pca[c_ind,f2].components_.T)*np.identity(20)))
#             k = k +1 
            
            
# fig, ax = plt.subplots(figsize = (10,10))

# ax.imshow(Overlap, cmap='viridis')

# %% Trajectories of task variables on different rules

# %% Archive, subspace overlap

    
# for c_ind in c_list:
#     for f in np.arange(ax_sz):
#         V_cap1 = 1-np.linalg.norm(Convdata[c_ind,f].T- 
#                                   np.dot(np.dot(Convdata[c_ind,f].T,pca[c_ind,f].components_.T),
#                                   pca[c_ind,f].components_))/np.linalg.norm(Convdata[c_ind,f].T)
        
#         for f2 in np.arange(ax_sz):
#             V_cap2 = 1-np.linalg.norm(Convdata[c_ind,f].T- 
#                                       np.dot(np.dot(Convdata[c_ind,f].T,pca[c_ind,f2].components_.T),
#                                       pca[c_ind,f2].components_))/np.linalg.norm(Convdata[c_ind,f].T)
            
#             Overlap[c_ind][f,f2] = V_cap2/V_cap1

     
# for f in np.arange(ax_sz):
#     V_cap1 = 1-np.linalg.norm(Convdata[c_list[1],f].T- 
#                               np.dot(np.dot(Convdata[c_list[1],f].T,pca[c_list[1],f].components_.T),
#                               pca[c_list[1],f].components_))/np.linalg.norm(Convdata[c_list[1],f].T)
        
#     for f2 in np.arange(ax_sz):
#         V_cap2 = 1-np.linalg.norm(Convdata[c_list[1],f].T- 
#                                   np.dot(np.dot(Convdata[c_list[1],f].T,pca[c_list[0],f2].components_.T),
#                                   pca[c_list[0],f2].components_))/np.linalg.norm(Convdata[c_list[1],f].T)
            
#         Overlap_across[f,f2] = V_cap2/V_cap1


sizeX = 0
p = 0

for n in good_list:
    n = int(n)
    X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,-2)
    if sizeX != np.size(X,0):        
        fig, ax = plt.subplots(1,2,figsize = (10,5))
        
        X1, Y1, L1, Rt1 = import_data_w_Ca(D_ppc,n,prestim,t_period,window,-1)

        
        ax[0].plot(np.mean(L1,0)*20)
        ax[1].plot(np.mean(L,0)*20)
        sizeX = np.size(X,0)
        p += 1