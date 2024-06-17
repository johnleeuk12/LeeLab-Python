# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:27:31 2024

Decoder analysis for each segments of 50 trials.



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
# fname = 'CaData_all_withlicktime_correctedv2.mat'


fname = 'CaData_all_all_session_v2_corrected.mat'
# fname = 'CaData_PIC_2.mat'


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
    # Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
    # Xpre = Xpre[:,None]
    
    # # 10/05 adding stim history, inverting sign for 10kHz = Go
    # Xprestim = np.concatenate(([0],-1*X[0:-1,3]+1),0)
    # Xprestim = Xprestim[:,None]
    # # Add reward instead of action
    # # currently it's Contingency, Stimulus, Reward, and Reward History
    # # X2 = np.column_stack([X[:,0],X[:,3],
    # #                       X[:,2]*X[:,1],Xpre]) 

    # # X2 = np.column_stack([Xprestim,-1*X[:,3]+1,
    # #                       X[:,2]*X[:,1],Xpre]) 
    
    # XHit = (X[:,0]==1)*(X[:,1]==1)
    # XFA  = (X[:,0]==0)*(X[:,1]==1)
    # Xmiss = (X[:,0]==1)*(X[:,1]==0)
    # XCR = (X[:,0]==0)*(X[:,1]==0)
    
    # X2 = np.column_stack([XHit,Xmiss,XFA,XCR])
    # X2 = np.row_stack([[0,0,0,0],X2[:-1,:]]) # previous hit FA miss etc
    # X2 = np.column_stack([X2,XHit,Xmiss,XFA,XCR])
    
    # Adding Lick(action as well)
    # X2 = np.column_stack([Xprestim,-1*X[:,3]+1,X[:,1],
    #                       X[:,2]*X[:,1],Xpre]) 


    
    return X,Y, L2, Rt

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

window = 100 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 10 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [3]
ModelW = {}



if ca ==0:
    D_ppc = load_matfile()
    # good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    good_list = find_good_data_Ca(t_period)
    
    


# %% building SVM with Ca data 

# using 4FA and 4CR trials for training and equal number for testing
ntr1 = 20 # train
ntr2 = 20 # test
# ind = 0
c_ind = 3

good_list_sep = np.arange(np.size(D_ppc,0))
# PAC_list = good_list_sep[good_list_sep<180] # PPCIC
PAC_list = good_list_sep[good_list_sep>180] 
f = 0
ylens = int((t_period+prestim)/window)

Xm = np.zeros((np.size(PAC_list),ntr1*2))
Xt = np.zeros((np.size(PAC_list),ntr2*2))
cv = 10
Acc = np.zeros ((cv,ylens))
Accshuffle= np.zeros ((cv,ylens))
x_axis = np.arange(1, t_period+prestim, window)
Acc_all = {}


for k in np.arange(cv):
    ModelW[c_ind,k] = np.zeros((np.size(PAC_list),ylens))
# from sklearn.model_selection import train_test_split
from sklearn import svm

t_ind = 2 # Hit

Data = {}
for n in PAC_list:
    n = int(n)
    X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
    l = {}
    # l[0] = np.concatenate([np.argwhere((X[:,3]==1)),np.argwhere((X[:,0]==1))]) # Current Correct(Check if previous)
    # l[1] = np.concatenate([np.argwhere((X[:,1]==1)),np.argwhere((X[:,2]==1))])# Currect IC
    l[0] = np.argwhere((X[:,0]==0)) #No-Go
    l[1] = np.argwhere((X[:,0]==1)) #Go
    Data[n] = {"X":X,"l" : l,"Y":Y}


for t in np.arange(ylens):
    if np.mod(t,5) == 0:
        print("print :  {} /80" .format(t))
        
    
    for k in np.arange(cv):
        tt = 0
        Xm = np.zeros((np.size(PAC_list),ntr1*2))
        Xt = np.zeros((np.size(PAC_list),ntr2*2))
        for n in PAC_list:
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
        ModelW[c_ind,k][:,t] = clf.coef_
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


# %% projection and decoding accuracy


def sigmoid(x):
    c1 = 5
    c2 = 0.5   
    return 1 / (1 + np.exp(-c1*(x-c2)))

Data2 = {}
ntr3 = 30
c_ind = 1 # decoder
c_ind2 = 2 # compare

Xt2 = np.zeros((np.size(PAC_list),ntr3*2))
# Xp1 = np.zeros((np.size(PAC_list),ylens))
# Xp2 = np.zeros((np.size(PAC_list),ylens))
# Yp1 = np.zeros((cv,ylens))
# Yp2 = np.zeros((cv,ylens))
ModelWmean = np.zeros((np.size(PAC_list),ylens))
decoder = {} # decoder time period
t1 = 40
t2 = 50

pre_t1 = 10
# pre_t2


for k in np.arange(cv):
    ModelWmean = ModelWmean + ModelW[c_ind,k]
    decoder[k]= np.mean(ModelW[c_ind,k][:,t1:t2],1)

ModelWmean = ModelWmean/cv


# decoder = np.mean(ModelWmean[:,t1:t2],1)
Acc2 = np.zeros ((cv,ylens))

for n in PAC_list:
    n = int(n)
    X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind2)
    l = {}

    l[0] = np.argwhere((X[:,0]==0)) # No-Go
    l[1] = np.argwhere((X[:,0]==1)) # Go
    Data2[n] = {"X":X,"l" : l,"Y":Y}


for t in np.arange(ylens):
    if np.mod(t,5) == 0:
        print("print :  {} /80" .format(t))
    for k in np.arange(cv):
        tt = 0
        # Xt2 = np.zeros((np.size(PAC_list[ind,f]),ntr3*2))
        for n in PAC_list:
            X = Data2[n]["X"]
            l = Data2[n]["l"]
            Y = Data2[n]["Y"]
            if np.size(l[0],0)>=ntr3 and np.size(l[1],0)>=ntr3:
                b = np.random.choice(l[0].ravel(),ntr3,replace = False)
                Xt2[tt,0:ntr3] = Y[b,t]-np.median(Y[:,:])
                    
                c = np.random.choice(l[1].ravel(),ntr3,replace = False)
                Xt2[tt,ntr3:ntr3*2] = Y[c,t]-np.median(Y[:,:])
            tt += 1
        Yt2= np.concatenate((np.zeros((ntr3,1)),np.ones((ntr3,1))))
        Yt2 = Yt2.ravel()
        Yhat = sigmoid(decoder[k] @ Xt2)

        # Yp1[k,t] = Xp1[:,t] @ decoder[k]
        # Yp2[k,t] = Xp2[:,t] @ decoder[k]
        Acc2[k,t] = 1-np.sum(np.abs(Yhat-Yt2))/(ntr3*2)




y1 = np.mean(ndimage.gaussian_filter(Acc2,[0,1]),0)
s1 = np.std(ndimage.gaussian_filter(Acc2,[0,1]),0)# /np.sqrt(cv)

# y2 = np.mean(ndimage.gaussian_filter(Accshuffle,[0,1]),0)
# s2 = np.std(ndimage.gaussian_filter(Accshuffle,[0,1]),0)#/np.sqrt(cv)   #/np.sqrt(8)
fig, axes = plt.subplots(1,1,figsize = (10,8))
axes.plot(x_axis*1e-3-prestim*1e-3,y1,c = 'blue')
axes.fill_between(x_axis*1e-3-prestim*1e-3,y1-s1,y1+s1,facecolor = 'blue',alpha = 0.3)        
axes.set_ylim([0.45,1])
            
        
# y1 = np.mean(ndimage.gaussian_filter(Yp1,[0,1]),0)
# s1 = np.std(ndimage.gaussian_filter(Yp1,[0,1]),0)# /np.sqrt(cv)

# y2 = np.mean(ndimage.gaussian_filter(Yp2,[0,1]),0)
# s2 = np.std(ndimage.gaussian_filter(Yp2,[0,1]),0)# /np.sqrt(cv)
        
        
# fig, axes = plt.subplots(1,1,figsize = (10,8))
# axes.plot(x_axis*1e-3-prestim*1e-3,y1,c = 'blue')
# axes.fill_between(x_axis*1e-3-prestim*1e-3,y1-s1,y1+s1,facecolor = 'blue',alpha = 0.3)     

# axes.plot(x_axis*1e-3-prestim*1e-3,y2,c = 'red')
# axes.fill_between(x_axis*1e-3-prestim*1e-3,y2-s2,y2+s2,facecolor = 'red',alpha = 0.3)    


# %% Decoder correlation
Corr = {}
decoder = {}
t1 = 20
t2 = 25

pre_t1 = 10
# pre_t2

for c_ind in [1,2]:
    for k in np.arange(cv):
        decoder[c_ind,k]= np.mean(ModelW[c_ind,k][:,t1:t2],1)
        

for k in np.arange(cv):
    Corr[k] = np.corrcoef((decoder[1,k],decoder[2,k]))
    






