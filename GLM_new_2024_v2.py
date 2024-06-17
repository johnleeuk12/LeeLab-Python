# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:28:11 2024

GLM analysis, without segmenting data by trial onset time. 


@author: Jong Hoon Lee
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


# %% File name and directory

# change fname for filename
# fname = 'CaData_all_all_session_v2_corrected.mat'
fname = 'CaData_all_session_v3_corrected.mat'

fdir = 'D:\Python\Data'


# %% Helper functions for loading and selecting data
np.seterr(divide = 'ignore') 
def load_matfile(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_dataset']
    return D_ppc 

def load_matfile_Ca(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_CaData']
    return D_ppc 



def import_data_w_Ca(D_ppc,n,window,c_ind):    
    # For each neuron, get Y, neural data and X task variables.  
    # Stim onset is defined by stim onset time
    # Reward is defined by first lick during reward presentation
    # Lick onset, offset are defined by lick times
    # Hit vs FA are defined by trial conditions
    
    

    N_trial = np.size(D_ppc[n,2],0)
    
    # extracting licks, the same way
    

    
    
    
    ### Extract Ca trace ###
    Yraw = {}
    Yraw = D_ppc[n,0]
    time_point = D_ppc[n,3]*1e3
    t = 0
    time_ind = []
    while t*window < np.max(time_point):
        time_ind = np.concatenate((time_ind,np.argwhere(time_point[0,:]>t*window)[0]))
        t += 1
    
    
    Y = np.zeros((1,len(time_ind)-1))   
    for t in np.arange(len(time_ind)-1):
        Y[0,t] = np.mean(Yraw[0,int(time_ind[t]):int(time_ind[t+1])])
        
    
    ### Extract Lick ### 
    L_all = np.zeros((1,len(time_ind)-1))
    L_all_onset = np.zeros((1,len(time_ind)-1))
    L_all_offset = np.zeros((1,len(time_ind)-1))
    # Rt = np.zeros((1,len(time_ind)-1))
    Ln = np.array(D_ppc[n,1])
    InterL = Ln[1:,:]- Ln[:-1,:]
    lick_onset= np.where(InterL[:,0]>2)[0] # lick bout boundary =2
    lick_onset = lick_onset+1
    lick_offset = lick_onset-1
    
    for l in np.floor(D_ppc[n,1]*(1e3/window)): 
        L_all[0,int(l[0])-1] = 1 
            
    for l in np.floor(Ln[lick_onset,0]*(1e3/window)):
        L_all_onset[0,int(l)-1] = 1
    
    for l in np.floor(Ln[lick_offset,0]*(1e3/window)):
        L_all_offset[0,int(l)-1] = 1 
            
    # for l in np.floor(D_ppc[n,6][:,0]*(1e3/window)):
    #     Rt[0,int(l)-1] = 1     
    
    ### Extract Lick End ###
    

    stim_onset =np.round(D_ppc[n,3][0,D_ppc[n,2][:,0]]*(1e3/window))
    stim_onset = stim_onset.astype(int)
    Rt = np.floor(D_ppc[n,6][:,0]*(1e3/window))
    Rt =Rt.astype(int)

    X = D_ppc[n,2][:,2:6] # task variables
    Xstim = X[:,0]
    XHit = (X[:,0]==1)*(X[:,1]==1)
    XFA  = (X[:,0]==0)*(X[:,1]==1)
    Xmiss = (X[:,0]==1)*(X[:,1]==0)
    XCR = (X[:,0]==0)*(X[:,1]==0)
    
    
    ### Create variables ###
    ED1 = 5 # 500ms pre, 1second post lag
    ED2 = 10
    stim_dur = 5 # 500ms stim duration
    delay = 10 # 1 second delay
    r_dur = 10 # 2 second reward duration 
    ED3 = 30 # 4 seconds post reward lag
    ED4 = 70
    ED_hist1 = 50 # 4 seconds pre-stim next trial
    ED_hist2 = 15 # 1.5 seconds post-stim next trial
    h_dur = 5
    
    X3_Lick_onset = np.zeros((ED1+ED2+1,np.size(Y,1)))
    X3_Lick_offset = np.zeros_like(X3_Lick_onset)
    
    X3_Lick_onset[0,:] = L_all_onset
    X3_Lick_offset[0,:] = L_all_offset

    for lag in np.arange(ED1):
        X3_Lick_onset[lag+1,:-lag-1] = L_all_onset[0,lag+1:]
        X3_Lick_offset[lag+1,:-lag-1] = L_all_offset[0,lag+1:]
    
    for lag in np.arange(ED2):
        X3_Lick_onset[lag+ED1+1,lag+1:] = L_all_onset[0,:-lag-1]
        X3_Lick_offset[lag+ED1+1,lag+1:] = L_all_offset[0,:-lag-1]
        
      
    X3_go = np.zeros((ED2+1,np.size(Y,1)))
    X3_ng = np.zeros_like(X3_go)
    
    for st in stim_onset[(Xstim == 1)]:
        X3_go[0,st:st+stim_dur] = 1
    
    for st in stim_onset[(Xstim ==0)]:
        X3_ng[0,st:st+stim_dur] = 1
        
    for lag in np.arange(ED2):
        X3_go[lag+1,lag+1:] = X3_go[0,:-lag-1]
        X3_ng[lag+1,lag+1:] = X3_ng[0,:-lag-1]
    
    
    X3_Hit = np.zeros((ED4+1,np.size(Y,1)))
    X3_FA = np.zeros_like(X3_Hit)
    X3_Miss = np.zeros_like(X3_Hit)
    X3_CR = np.zeros_like(X3_Hit)
    
    # X3_Miss = np.zeros((ED3+1,np.size(Y,1)))
    # X3_CR = np.zeros_like(X3_Miss)
    # for r in Rt[(XHit == 1)]:
    #     if r != 0:
    #         X3_Hit[0,r:r+r_dur] = 1
    
    # for r in Rt[(XFA == 1)]:
    #     if r != 0:
    #         X3_FA[0,r:r+r_dur] = 1
    for r in Rt[(XHit == 1)]:
        if r != 0:
            r = r-10
            X3_Hit[0,r:r+r_dur] = 1
    
    for r in Rt[(XFA == 1)]:
        if r != 0:
            r = r-10
            X3_FA[0,r:r+r_dur] = 1
            
    for st in stim_onset[(Xmiss ==1)]:        
        X3_Miss[0,st+delay+stim_dur : st+delay+stim_dur+r_dur] = 1

    for st in stim_onset[(XCR ==1)]:        
        X3_CR[0,st+delay+stim_dur : st+delay+stim_dur+r_dur] = 1 
        
        
    # for lag in np.arange(ED3):
    #     # X3_Hit[lag+1,lag+1:] = X3_Hit[0,:-lag-1]
    #     # X3_FA[lag+1,lag+1:] = X3_FA[0,:-lag-1]
    #     X3_Miss[lag+1,lag+1:] = X3_Miss[0,:-lag-1]
    #     X3_CR[lag+1,lag+1:] = X3_CR[0,:-lag-1]

    for lag in np.arange(ED4):
        X3_Miss[lag+1,lag+1:] = X3_Miss[0,:-lag-1]
        X3_CR[lag+1,lag+1:] = X3_CR[0,:-lag-1]
        X3_Hit[lag+1,lag+1:] = X3_Hit[0,:-lag-1]
        X3_FA[lag+1,lag+1:] = X3_FA[0,:-lag-1]

    # X3_Hit_hist = np.zeros((ED_hist1+ED_hist2+1,np.size(Y,1)))
    # X3_FA_hist = np.zeros((ED_hist1+ED_hist2+1,np.size(Y,1)))
    
    X3_Hit_hist = np.zeros((ED_hist1+1,np.size(Y,1)))
    X3_FA_hist = np.zeros((ED_hist1++1,np.size(Y,1)))
    # XHit_prev = np.concatenate(([False], XHit[0:-1]), axis = 0)
    # XFA_prev = np.concatenate(([False], XFA[0:-1]), axis = 0)
    
    
    # X3_Hit_hist[0,30:] = X3_Hit[0,:-30]
    # X3_FA_hist[0,30:] = X3_FA[0,:-30]
    
    # for lag in np.arange(ED_hist1):
    #     X3_Hit_hist[lag+1,lag+1:] = X3_Hit_hist[0,:-lag-1]
    #     X3_FA_hist[lag+1,lag+1:] = X3_FA_hist[0,:-lag-1]
    # for st in stim_onset[(XHit_prev ==1)]:
    #     X3_Hit_hist[0,st:st+h_dur] = 1
    # for st in stim_onset[(XFA_prev ==1)]:
    #     X3_FA_hist[0,st:st+h_dur] = 1 
    
    
    # for lag in np.arange(ED_hist1):
    #     X3_Hit_hist[lag+1,:-lag-1] = X3_Hit_hist[0,lag+1:]
    #     X3_FA_hist[lag+1,:-lag-1] = X3_FA_hist[0,lag+1:]
    
    # for lag in np.arange(ED_hist2):
    #     X3_Hit_hist[lag+ED_hist1+1,lag+1:] = X3_Hit_hist[0,:-lag-1]
    #     X3_FA_hist[lag+ED_hist1+1,lag+1:] = X3_FA_hist[0,:-lag-1]
    
    # for r in Rt[()]
    # gather X variables
    
    
    X3 = {}
    X3[0] = X3_Lick_onset
    X3[1] = X3_Lick_offset
    X3[2] = X3_go
    X3[3] = X3_ng
    X3[4] = X3_Hit
    X3[5] = X3_FA
    X3[6] = X3_Miss
    X3[7] = X3_CR
    X3[8] = X3_Hit_hist
    X3[9] = X3_FA_hist
    
    if c_ind == 1:
        c1 = stim_onset[1]-100
        c2 = stim_onset[151]
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[1:150]
        r_onset = r_onset[1:150]
        Xstim = Xstim[1:150]
    
    elif c_ind == 3:
        c1 = stim_onset[200]-100
        c2 = stim_onset[D_ppc[n,4][0][0]+26] 
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[200:D_ppc[n,4][0][0]+26]
        r_onset = r_onset[200:D_ppc[n,4][0][0]+26]
        Xstim = Xstim[200:D_ppc[n,4][0][0]+26]

    elif c_ind == 2:       
        c1 = stim_onset[D_ppc[n,4][0][0]+26]-100 
        c2 = stim_onset[399]
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[D_ppc[n,4][0][0]+26:398]
        r_onset = r_onset[D_ppc[n,4][0][0]+26:398]
        Xstim = Xstim[D_ppc[n,4][0][0]+26:398]

        

    Y = Y[:,c1:c2]
    L_all = L_all[:,c1:c2]
    L_all_onset = L_all_onset[:,c1:c2]
    L_all_offset = L_all_offset[:,c1:c2]
    
    Y0 = np.mean(Y[:,:stim_onset[1]-50])
    
    for ind in np.arange(len(X3)):
        X3[ind] = X3[ind][:,c1:c2]         



    return X3,Y, L_all,L_all_onset, L_all_offset, stim_onset2, r_onset, Xstim,Y0

# %% glm_per_neuron function code
def glm_per_neuron(n,c_ind, fig_on):
    X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim, Y0 = import_data_w_Ca(D_ppc,n,window,c_ind)
    
    Y2 = Y #-Y0
    # X4 = np.ones((1,np.size(Y)))
    
    reg = ElasticNet(alpha = 4*1e-2, l1_ratio = 0.5, fit_intercept=True) #Using a linear regression model with Ridge regression regulator set with alpha = 1
    ss= ShuffleSplit(n_splits=k, test_size=0.30, random_state=0)
    
    ### initial run, compare each TV ###
    Nvar= len(X) -2
    compare_score = {}
    int_alpha = 10
    for a in np.arange(Nvar+1):
        
        # X4 = np.ones_like(Y)*int_alpha
        X4 = np.zeros_like(Y)

        if a < Nvar:
            X4 = np.concatenate((X4,X[a]),axis = 0)

        cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                    return_estimator = True, 
                                    scoring = 'r2') 
        compare_score[a] = cv_results['test_score']
    
    f = np.zeros((1,Nvar))
    p = np.zeros((1,Nvar))
    score_mean = np.zeros((1,Nvar))
    for it in np.arange(Nvar):
        f[0,it], p[0,it] = stats.ks_2samp(compare_score[it],compare_score[Nvar],alternative = 'less')
        score_mean[0,it] = np.mean(compare_score[it])

    max_it = np.argmax(score_mean)
    init_score = compare_score[max_it]
    init_compare_score = compare_score
    
    if p[0,max_it] > 0.05:
            max_it = []
    else:  
            # === stepwise forward regression ===
            step = 0
            while step < Nvar:
                max_ind = {}
                compare_score2 = {}
                f = np.zeros((1,Nvar))
                p = np.zeros((1,Nvar))
                score_mean = np.zeros((1,Nvar))
                for it in np.arange(Nvar):
                    m_ind = np.unique(np.append(max_it,it))
                    # X4 = np.ones_like(Y)*int_alpha
                    X4 = np.zeros_like(Y)
                    for a in m_ind:
                        X4 = np.concatenate((X4,X[a]),axis = 0)

                    
                    cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                                return_estimator = True, 
                                                scoring = 'r2') 
                    compare_score2[it] = cv_results['test_score']
    
                    f[0,it], p[0,it] = stats.ks_2samp(compare_score2[it],init_score,alternative = 'less')
                    score_mean[0,it] = np.mean(compare_score2[it])
                max_ind = np.argmax(score_mean)
                if p[0,max_ind] > 0.05 or p[0,max_ind] == 0:
                    step = Nvar
                else:
                    max_it = np.unique(np.append(max_it,max_ind))
                    init_score = compare_score2[max_ind]
                    step += 1
                    
            # === forward regression end ===
            
            # === running regression with max_it ===
            
            # X4 = np.ones_like(Y)*int_alpha
            X4 = np.zeros_like(Y)
            if np.size(max_it) == 1:
                X4 = np.concatenate((X4,X[max_it]),axis = 0)
            else:
                for a in max_it:
                    X4 = np.concatenate((X4,X[a]),axis = 0)
            
            cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                        return_estimator = True, 
                                        scoring = 'r2') 
            score3 = cv_results['test_score']
            
            theta = [] 
            inter = []
            yhat = []
            for model in cv_results['estimator']:
                theta = np.concatenate([theta,model.coef_]) 
                # inter = np.concatenate([inter, model.intercept_])
                yhat =np.concatenate([yhat, model.predict(X4.T)])
                
            theta = np.reshape(theta,(k,-1)).T
            yhat = np.reshape(yhat,(k,-1)).T
            yhat = yhat + Y0
    
    TT = {}
    lg = 1
    
    if np.size(max_it) ==1:
        a = np.empty( shape=(0, 0) )
        max_it = np.append(a, [int(max_it)]).astype(int)
    try:
        for t in max_it:
            TT[t] = X[t].T@theta[lg:lg+np.size(X[t],0),:]  
            lg = lg+np.size(X[t],0)
    except: 
        TT[max_it] = X[max_it].T@theta[lg:lg+np.size(X[max_it],0),:]  
    
    if 4 in max_it:
        if 8 in max_it:
            TT[4] = TT[4] + TT[8]
    elif 8 in max_it:
        TT[4] = TT[8]
        max_it = np.append(max_it, [4])
    
        
    if 5 in max_it:
        if 9 in max_it:
            TT[5] = TT[5] + TT[9]
    elif 9 in max_it:
        TT[5] = TT[9]
        max_it = np.append(max_it, [5])
        
    
    # === figure === 
    if fig_on ==1:
        prestim = 20
        t_period = 60
        
        y = np.zeros((t_period+prestim,np.size(stim_onset)))
        yh = np.zeros((t_period+prestim,np.size(stim_onset)))
        l = np.zeros((t_period+prestim,np.size(stim_onset))) 
        weight = {}
        for a in np.arange(Nvar):
           weight[a] = np.zeros((t_period+prestim,np.size(stim_onset))) 
        
        yhat_mean = np.mean(yhat,1).T - Y0    
        for st in np.arange(np.size(stim_onset)):
            y[:,st] = Y[0,stim_onset[st]-prestim: stim_onset[st]+t_period]
            yh[:,st] = yhat_mean[stim_onset[st]-prestim: stim_onset[st]+t_period]
            l[:,st] = Lm[0,stim_onset[st]-prestim: stim_onset[st]+t_period]
            # if np.size(max_it)>1:
            for t in max_it:
                weight[t][:,st] = np.mean(TT[t][stim_onset[st]-prestim: stim_onset[st]+t_period,:],1)
            # else:
            #     weight[max_it][:,st] = np.mean(TT[max_it][stim_onset[st]-prestim: stim_onset[st]+t_period,:],1)
            
    
        
        xaxis = np.arange(t_period+prestim)- prestim
        xaxis = xaxis*1e-1
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))        
        cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
        clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR','Hit_pre','FA_pre']
        lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted']
        
        ### plot y and y hat
        stim_ind1 = (Xstim ==1)
        stim_ind2 = (Xstim ==0)
    
        y1 = ndimage.gaussian_filter(np.mean(y[:,stim_ind1],1),0)
        y2 = ndimage.gaussian_filter(np.mean(y[:,stim_ind2],1),0)
        s1 = np.std(y[:,stim_ind1],1)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(y[:,stim_ind2],1)/np.sqrt(np.sum(stim_ind2))
        
        ax1.plot(xaxis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y1-s1, y1+s1, color = "blue",alpha = 0.5)
        ax1.plot(xaxis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y2-s2, y2+s2, color = "red",alpha = 0.5)
        
        y1 = ndimage.gaussian_filter(np.mean(yh[:,stim_ind1],1),0)
        y2 = ndimage.gaussian_filter(np.mean(yh[:,stim_ind2],1),0)
        s1 = np.std(yh[:,stim_ind1],1)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(yh[:,stim_ind2],1)/np.sqrt(np.sum(stim_ind2))
        
        ax1.plot(xaxis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y1-s1, y1+s1, color = "gray",alpha = 0.5)
        ax1.plot(xaxis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y2-s2, y2+s2, color = "gray",alpha = 0.5)
        
        
        
        ### plot model weights
        for a in np.arange(Nvar):
            y1 = ndimage.gaussian_filter(np.mean(weight[a],1),0)
            s1 = np.std(weight[a],1)/np.sqrt(np.size(weight[a],1))
            
            
            ax2.plot(xaxis,ndimage.gaussian_filter(y1,1),linewidth = 2.0,
                     color = cmap[a], label = clabels[a], linestyle = lstyles[a])
            ax2.fill_between(xaxis,(ndimage.gaussian_filter(y1,1) - s1),
                            (ndimage.gaussian_filter(y1,1)+ s1), color=cmap[a], alpha = 0.2)
        
        ### plot lick rate ###
        
        y1 = ndimage.gaussian_filter(np.mean(l[:,stim_ind1],1),0)
        y2 = ndimage.gaussian_filter(np.mean(l[:,stim_ind2],1),0)
        s1 = np.std(l[:,stim_ind1],1)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(l[:,stim_ind2],1)/np.sqrt(np.sum(stim_ind2))
        
        ax3.plot(xaxis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = 'solid')
        ax3.fill_between(xaxis,y1-s1, y1+s1, color = "blue",alpha = 0.5)
        ax3.plot(xaxis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax3.fill_between(xaxis,y2-s2, y2+s2, color = "red",alpha = 0.5)
        
        
        ax2.set_title('unit_'+str(n+1))
        sc = np.mean(score3)
        ax4.set_title(f'{sc:.2f}')
        plt.show()
    
    
    return Xstim, L_on, inter, TT, Y, max_it, score3, init_compare_score, yhat



    
    
    
# %% Initialize
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

window = 100 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 20 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [3]



if ca ==0:
    D_ppc = load_matfile()
    # good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    # good_list = find_good_data_Ca(t_period)
    
    
    
    
# %% Run GLM

Data = {}



for c_ind in c_list:
    # t = 0 
    good_list2 = [];
    for n in np.arange(np.size(D_ppc,0)):
        
        n = int(n)
        if D_ppc[n,4][0][0] > 0:
            try:
                Xstim, L_on, inter, TT, Y, max_it, score3, init_score, yhat  = glm_per_neuron(n,c_ind,1)
                Data[n,c_ind-1] = {"X":Xstim,"coef" : TT, "score" : score3, 'Y' : Y,'init_score' : init_score,
                                    "intercept" : inter,'L' : L_on,"yhat" : yhat}
                good_list2 = np.concatenate((good_list2,[n]))
                print(n)
                
            except KeyboardInterrupt:
                
                break
            except:
            
                print("Break, no fit") 
# np.save('RT2new_0423.npy', Data,allow_pickle= True)     
# Data2 = np.load('RTnew_0411.npy',allow_pickle= True).item()
# test = Data2.item()

# test1 =test(7,2)
  # %% plot R score 


d_list3 = good_list2 <= 195
# d_list3 = good_list <= 118

d_list = good_list2 > 195
cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
Sstyles = ['tab:orange','none','tab:blue','none','tab:red','none','black','green','tab:purple','none']


def make_RS(d_list):
    good_list_sep = good_list2[d_list]
    ax_sz = len(cmap)-2
    I = np.zeros((np.size(good_list_sep),ax_sz+1))
       
        
    for n in np.arange(np.size(good_list_sep,0)):
        nn = int(good_list_sep[n])
        # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
        Model_score = Data[nn, c_ind-1]["score"]
        init_score =  Data[nn, c_ind-1]["init_score"]
        for a in np.arange(ax_sz):
            I[n,a] = np.mean(init_score[a])
        I[n,ax_sz] = np.mean(Model_score)
        
    
    fig, axes = plt.subplots(1,1, figsize = (10,8))
        # Rsstat = {}
    for a in np.arange(ax_sz):
        Rs = I[:,a]
        Rs = Rs[Rs>0.01]
        axes.scatter(np.ones_like(Rs)*(a+(c_ind+1)*-0.3),Rs,facecolors=Sstyles[a], edgecolors= cmap[a])
        axes.scatter([(a+(c_ind+1)*-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')    
            # Rs = Rs/(Rmax+0.03)
            # Rsstat[c_ind,f] = Rs
    
                # axes.boxplot(Rs,positions= [f+(c_ind+1)*-0.3])
    Rs = I[:,ax_sz]
    Rs = Rs[Rs>0.02]
    axes.scatter(np.ones_like(Rs)*(ax_sz+(c_ind+1)*-0.3),Rs,c = 'k',)
    axes.scatter([(ax_sz+(c_ind+1)*-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')
    axes.set_ylim([0,0.75])
    axes.set_xlim([-1,len(cmap)])
    
    
    return I

I1 = make_RS(d_list3)
I2 = make_RS(d_list)

bins = np.arange(0,0.25, 0.005)
fig, axs= plt.subplots(1,1,figsize = (5,5))
axs.hist(np.max(I2,1),bins = bins,alpha = 0.8)
axs.hist(np.max(I1,1),bins = bins,alpha = 0.8)
# np.size(np.max(I1,1))

# %% histogram of TV encoding
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green']
edgec = cmap
# edgec = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']


d_list2 = d_list
def TV_hist(d_list2):
    good_list_sep = good_list2[d_list2]
    TV = np.empty([1,1])
    for n in np.arange(np.size(good_list_sep,0)):
            nn = int(good_list_sep[n])
            Model_coef = Data[nn, c_ind-1]["coef"]
            max_it = [key for key in Model_coef]
            TV = np.append(TV, max_it)
    
    TV = TV[1:]
    ax_sz = 8
    B = np.zeros((1,ax_sz))
    for f in np.arange(ax_sz):
        B[0,f] = np.sum(TV == f)
        
    B = B/np.sum(d_list2)
    fig, axes = plt.subplots(1,1, figsize = (15,5))
    axes.grid(visible=True,axis = 'y')
    axes.bar(np.arange(ax_sz)*3,B[0,:], color = "white", edgecolor = edgec, alpha = 1, width = 0.5, linewidth = 2,hatch = '/')
    # axes.bar(np.arange(ax_sz)*3,B[0,:], color = cmap, edgecolor = edgec, alpha = 1, width = 0.5, linewidth = 2,hatch = '/')
    axes.set_ylim([0,0.8])
            
TV_hist(d_list3)
        
# %% 


def extract_onset_times(D_ppc,n):
    window = 100
    stim_onset =np.round(D_ppc[n,3][0,D_ppc[n,2][:,0]]*(1e3/window))
    stim_onset = stim_onset.astype(int)
    Rt = np.floor(D_ppc[n,6][:,0]*(1e3/window))
    Rt =Rt.astype(int)
    
    
    if c_ind == 1:
        c1 = stim_onset[1]-100
        c2 = stim_onset[151]
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[1:150]
        r_onset = Rt-c1
        r_onset = r_onset[1:150]

    
    elif c_ind == 3:
        c1 = stim_onset[200]-100
        c2 = stim_onset[D_ppc[n,4][0][0]+26] 
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[200:D_ppc[n,4][0][0]+26]
        r_onset = Rt-c1
        r_onset = r_onset[200:D_ppc[n,4][0][0]+26]



    elif c_ind == 2:       
        c1 = stim_onset[D_ppc[n,4][0][0]+26]-100 
        c2 = stim_onset[399]
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[D_ppc[n,4][0][0]+26:398]
        r_onset = Rt-c1
        # r_onset = r_onset[200:D_ppc[n,4][0][0]+26]
        r_onset = r_onset[D_ppc[n,4][0][0]+26:398]

    return stim_onset2, r_onset

    
for n in np.arange(np.size(good_list2,0)):
    nn = int(good_list2[n])
    # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
    stim_onset,r_onset = extract_onset_times(D_ppc,nn)
    Data[nn,c_ind-1]["stim_onset"] = stim_onset
    Data[nn,c_ind-1]["r_onset"] = r_onset

    # %% Normalized population average of task variable weights
# c_ind = 1
d_list = good_list2 > 195
# d_list3 = good_list <= 179
d_list3 = good_list2 <= 195

# Lic = np.where(good_listRu <180)
# Lic = Lic[0][-1]
# good_list_sep = good_listRu[:]

good_list_sep = good_list2[:]

weight_thresh = 5*1e-2


# fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))        
cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
# clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR']
lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted']
ax_sz = len(cmap)


Convdata = {}
pre = 10 # 10 40 
post = 70 # 50 20
xaxis = np.arange(post+pre)- pre
xaxis = xaxis*1e-1

for a in np.arange(ax_sz):
    Convdata[a] = np.zeros((np.size(good_list_sep),pre+post))

for n in np.arange(np.size(good_list_sep,0)):
    nn = int(good_list_sep[n])
    # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
    Model_coef = Data[nn, c_ind-1]["coef"]
    Model_score = Data[nn, c_ind-1]["score"]
    stim_onset2 =  Data[nn, c_ind-1]["stim_onset"]
    stim_onset =  Data[nn, c_ind-1]["stim_onset"]
    
    
    weight = {}
    max_it = [key for key in Model_coef]
    for a in max_it:
        weight[a] = np.zeros((pre+post,np.size(stim_onset))) 
        
            
    for st in np.arange(np.size(stim_onset)):
        for t in max_it:
            if stim_onset[st] <0:
                stim_onset[st] = stim_onset2[st]+15
                
            weight[t][:,st] = np.mean(Model_coef[t][stim_onset[st]-pre: stim_onset[st]+post,:],1)

    for a in max_it:    
        Convdata[a][n,:] = np.mean(weight[a],1) /(np.max(np.abs(np.mean(weight[a],1)))+0.2)
        
fig, axes = plt.subplots(1,1,figsize = (10,8))       
for a in np.arange(ax_sz):
    error = np.std(Convdata[a],0)/np.sqrt(np.size(good_list_sep))
    y = ndimage.gaussian_filter(np.mean(Convdata[a],0),2)
    y = np.abs(y)
    axes.plot(xaxis,y,c = cmap[a],linestyle = lstyles[a])
    axes.fill_between(xaxis,y-error,y+error,facecolor = cmap[a],alpha = 0.3)
    axes.set_ylim([-0.01,0.25])

# %% plot example neuron v2

# n = 321
pre = 20
post = 60
# f1 = 3
# f2 = 2
# t1 = 90
# t2 = 3
t1 = 93
t2 = 2
def plt_ex_neurons2(n,pre,post,t1,t2):
    fig, ax = plt.subplots(1,1, figsize= (30,5))
    stim_onset = Data[n, c_ind-1]["stim_onset"]
    Model_coef = Data[n, c_ind-1]["coef"]
    xaxis= np.arange(stim_onset[t1]-pre,stim_onset[t1+t2]+post)
    ax.plot(xaxis,Data[n,c_ind-1]["Y"][0,xaxis], color = "black", linewidth = 2)
    ax.vlines(stim_onset[t1:t1+t2+1],0,30)
    h = Data[n,c_ind-1]["yhat"][xaxis,:]
    yh = (np.mean(h,1))*2-1 
    sdh =np.std(h,1)
    ax.plot(xaxis,yh, color = "black", linestyle = "dashed",linewidth = 2)
    ax.fill_between(xaxis,yh-sdh,yh+sdh, alpha = 0.5)
    max_it = [key for key in Model_coef]
    cmap = ['black','grey','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green']
    
    xaxis= np.arange(stim_onset[t1]-pre,stim_onset[t1+t2]+post)
    
    for f in max_it:
        if f < 8:
            C = Model_coef[f][xaxis,:]
            yc = np.mean(C,1)*2
            sdc = np.std(C,1)*2
            ax.plot(xaxis,yc, color = cmap[f],linewidth = 3)
            ax.fill_between(xaxis,yc-sdc,yc+sdc,color = cmap[f], alpha = 0.5)
    
    plt.savefig("example neuron "+str(n)+ ".svg")

plt_ex_neurons2(126,pre,post,t1,t2)

# mY2 = np.zeros((20,70))
# t3 = 0
# for st in stim_onset[t1:t2+20]:
#     mY2[t3,:] = np.mean(Yh[st-10:st+60,:],1)
#     t3 += 1
# fig, ax = plt.subplots(1,1,figsize= (5,5))
# ax.plot(np.mean(mY,0))
# ax.plot(np.mean(mY2,0))



# stim_onset = Data[n, c_ind-1]["stim_onset"]
# Model_coef = Data[n, c_ind-1]["coef"]

# y = np.zeros((t_period+prestim,np.size(stim_onset)))
# yh = np.zeros((t_period+prestim,np.size(stim_onset)))
# l = np.zeros((t_period+prestim,np.size(stim_onset))) 
# prestim = 20
# t_period = 60
# weight = {}

# max_it = [key for key in Model_coef]
# for t in max_it:
#             weight[t] = np.zeros((t_period+prestim,np.size(stim_onset))) 
        
# yhat_mean = np.mean(yhat,1).T -np.mean(Y[0:1000])
# for st in np.arange(np.size(stim_onset)):
#     y[:,st] = Y[0,stim_onset[st]-prestim: stim_onset[st]+t_period]
#     yh[:,st] = yhat_mean[stim_onset[st]-prestim: stim_onset[st]+t_period]
#     # l[:,st] = Lm[0,stim_onset[st]-prestim: stim_onset[st]+t_period]
# # if np.size(max_it)>1:
#     for t in max_it:
#         weight[t][:,st] = np.mean(TT[t][stim_onset[st]-prestim: stim_onset[st]+t_period,:],1)

# # yhat_mean = np.mean(yhat,1).T -np.mean(Y[0:1000])

# # y = Y[0,stim_onset[t1]-prestim:stim_onset[t1+t2]+t_period]
# # yh = yhat_mean[stim_onset[t1]-prestim:stim_onset[t1+t2]+t_period]
# fig, ax = plt.subplots(1,1,figsize= (5,5))
# ax.plot(y[:,3])
# ax.plot(yh[:,3])
# %% plot example neuron

# n = 126
def plt_ex_neurons(nn,c1,c2):   
    # nn = int(good_list_sep[n])
    pre = 20
    post = 60
    prestim = pre*window
    t_period = post*window
    x_axis = np.arange(1, prestim+t_period, window)
    x_axis = (x_axis-prestim)*1e-3
    
    y_lens = np.arange(int((t_period+prestim)/window))    
    Y = Data[nn,c_ind-1]["Y"][:,:]
    # X = Data[nn,c_ind-1]["X"][:,:]
    Yhat = Data[nn,c_ind-1]["yhat"]
    intercept = Data[nn,c_ind-1]["intercept"]
    Model_coef = Data[nn, c_ind-1]["coef"]
    Model_score = Data[nn, c_ind-1]["score"]
    stim_onset2 =  Data[nn, c_ind-1]["stim_onset"]
    stim_onset =  Data[nn, c_ind-1]["stim_onset"]
    
    X4 = D_ppc[nn,2][:,2:6] # task variables

    X4 = X4[c1:c2,:]
    # ymean = np.ones((len(y_lens),np.size(X4,0))).T*intercept

    ### divide into Go, No Hit, Miss, FA and CR
    X5 = np.column_stack([(X4[:,0] == 1) * (X4[:,1] == 1), # Hit
                           (X4[:,0] == 1) * (X4[:,1] == 0), # MIss
                           (X4[:,0] == 0) * (X4[:,1] == 1), # FA
                           (X4[:,0] == 0) * (X4[:,1] == 0)]) # CR
    X5 = np.column_stack([X5,np.concatenate(([False],X5[:-1,0]),0),np.concatenate(([False],X5[:-1,1]),0)])
    X5 =X5[1:150,:]
    # stim_onset =np.round(D_ppc[nn,3][0,D_ppc[nn,2][:,0]]*(1e3/window))
    # stim_onset = stim_onset.astype(int)
    # stim_onset = stim_onset[0:149]
    # pooling model weights
    weight = {}
    Y2 = np.zeros((pre+post,np.size(stim_onset))) 
    Yhat2 = np.zeros((pre+post,np.size(stim_onset))) 
    max_it = [key for key in Model_coef]
    for a in np.arange(10):
        weight[a] = np.zeros((pre+post,np.size(stim_onset))) 
    for st in np.arange(np.size(stim_onset)):
        if stim_onset[st] <0:
            stim_onset[st] = stim_onset2[st]+15
        Y2[:,st] = Y[0,stim_onset[st]-pre: stim_onset[st]+post]
        Yhat2[:,st] = np.mean(Yhat[stim_onset[st]-pre: stim_onset[st]+post,:],1)
        for t in max_it:
            if stim_onset[st] <0:
                stim_onset[st] = stim_onset2[st]+15
                
            weight[t][:,st] = np.mean(Model_coef[t][stim_onset[st]-pre: stim_onset[st]+post,:],1)
    

    Y2 = Y2.T
    Yhat2 = Yhat2.T
    fig, axes = plt.subplots(6,6,figsize = (30,20))
    for ind1 in np.arange(6):
        axes[0,ind1].plot(x_axis,np.mean(Y2[X5[:,ind1],:],0),color = "blue",linewidth = 3.0)
        axes[0,ind1].plot(x_axis,np.mean(Yhat2[X5[:,ind1],:],0),color = "red",linewidth = 3.0)
        axes[0,ind1].xaxis.set_tick_params(labelsize=20)
        axes[0,ind1].yaxis.set_tick_params(labelsize=20)
        # axes[0,ind1].set_ylim([np.min(Yhat2)*1.5,np.max(Yhat2)*1.5])
    
    # pltmin = np.min([np.mean(yh[0],0),np.mean(yh[1],0),np.mean(yh[2],0),np.mean(yh[3],0)])
    # pltmax = np.max([np.mean(yh[0],0),np.mean(yh[1],0),np.mean(yh[2],0),np.mean(yh[3],0)])
    
    cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
    # clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR']
    lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted']
    for ind2 in np.arange(1,6):
        if ind2 ==1:
            yh1 = weight[2]
            yh2 = weight[3]
            color1 = cmap[2]
            color2 = cmap[3]
            ls1 = lstyles[2]
            ls2 = lstyles[3]
            pltmin = np.min([np.min(yh1),np.min(yh2)])
            pltmax = np.max([np.max(yh1),np.max(yh2)])
        elif ind2 ==2:
            yh1 = weight[4]
            yh2 = weight[7]
            color1 = cmap[4]
            color2 = cmap[7]
            ls1 = lstyles[4]
            ls2 = lstyles[7]
            if np.min([np.min(yh1),np.min(yh2)]) < pltmin:
                pltmin = np.min([np.min(yh1),np.min(yh2)])
            if np.max([np.max(yh1),np.max(yh2)]) > pltmax:
                pltmax = np.max([np.max(yh1),np.max(yh2)])
            
        elif ind2 ==3:
            yh1 = weight[5]
            yh2 = weight[6]
            color1 = cmap[5]
            color2 = cmap[6]
            ls1 = lstyles[5]
            ls2 = lstyles[6]
            if np.min([np.min(yh1),np.min(yh2)]) < pltmin:
                pltmin = np.min([np.min(yh1),np.min(yh2)])
            if np.max([np.max(yh1),np.max(yh2)]) > pltmax:
                pltmax = np.max([np.max(yh1),np.max(yh2)])
                
        elif ind2 ==4:
            yh1 = weight[0]
            yh2 = weight[1]    
            color1 = cmap[0]
            color2 = cmap[1]
            ls1 = lstyles[0]
            ls2 = lstyles[1]
            if np.min([np.min(yh1),np.min(yh2)]) < pltmin:
                pltmin = np.min([np.min(yh1),np.min(yh2)])
            if np.max([np.max(yh1),np.max(yh2)]) > pltmax:
                pltmax = np.max([np.max(yh1),np.max(yh2)])
                
        elif ind2 ==5:
            yh1 = weight[8]
            yh2 = weight[9]   
            color1 = cmap[8]
            color2 = cmap[9]
            ls1 = lstyles[8]
            ls2 = lstyles[9]
            if np.min([np.min(yh1),np.min(yh2)]) < pltmin:
                pltmin = np.min([np.min(yh1),np.min(yh2)])
            if np.max([np.max(yh1),np.max(yh2)]) > pltmax:
                pltmax = np.max([np.max(yh1),np.max(yh2)])
            
        
        for ind1 in np.arange(6):
            # yhat_tv = 
            # axes[ind2,ind1].plot(x_axis,np.mean(yh[ind2-2][X5[:,ind1],:],0)+np.mean(ymean,0),color = cmap[ind2-2])
            axes[ind2,ind1].plot(x_axis,np.mean(yh1[:,X5[:,ind1]],1),color = color1,linewidth = 3.0, linestyle = ls1)
            axes[ind2,ind1].plot(x_axis,np.mean(yh2[:,X5[:,ind1]],1),color = color2,linewidth = 3.0, linestyle = ls2)
            axes[ind2,ind1].set_ylim([pltmin*1.2,pltmax*1.2])
            axes[ind2,ind1].yaxis.set_tick_params(labelsize=20)
            axes[ind2,ind1].xaxis.set_tick_params(labelsize=20)
    return yh1, yh2   
    
    
nn = 126
c1 = 0
c2 = 200  
if c_ind == 3:
    c1 = 200
    c2 = D_ppc[nn,4][0][0]+26                            
yh1, yh2 = plt_ex_neurons(nn,c1,c2)  
# %% plotting weights by peak order
listOv = {}

f = 0
W5 = {}
W5AC= {}
W5IC = {}
max_peak3 ={}
tv_number = {}
b_count = {}
ax_sz = 8
for ind in [0,1]: # 0 is PPCIC, 1 is PPCAC
    b_count[ind] = np.zeros((2,ax_sz))

    for f in np.arange(ax_sz):
        W5[ind,f] = {}

for ind in [0,1]:
    for f in np.arange(ax_sz):
        list0 = (np.mean(Convdata[f],1) != 0)
        
        Lg = len(good_list2)
        Lic = np.where(good_list2 <194)
        Lic = Lic[0][-1]
        if ind == 0:
            list0[Lic:Lg] = False # PPCIC
        elif ind == 1:           
            list0[0:Lic] = False # PPCAC
        
        # list0ind = np.arange(Lg)
        # list0ind = list0ind[list0]
        list0ind = good_list2[list0]
        W = ndimage.uniform_filter(Convdata[f][list0,:],[0,0], mode = "mirror")
        
        max_peak = np.argmax(np.abs(W),1)
        max_ind = max_peak.argsort()
        
        list1 = []
        list2 = []
        list3 = []
        
        SD = np.std(W[:,:])
        for m in np.arange(np.size(W,0)):
            n = max_ind[m]
            # SD = np.std(W[n,:])
            # if SD< 0.05:
            #     SD = 0.05
            if max_peak[n]> 0:    
                if W[n,max_peak[n]] > 1*SD:
                    list1.append(m)
                    list3.append(m)
                elif W[n,max_peak[n]] <-1*SD:
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
        s ='+' + str(np.size(W1,0)) +  '-' + str(np.size(W2,0))
        print(s)
        b_count[ind][0,f] = np.size(W1,0)
        b_count[ind][1,f] = np.size(W2,0)
        W3 = np.concatenate((W1,-W2), axis = 0)
        tv_number[ind,f] = [np.size(W1,0),np.size(W2,0)]
        
        W5[ind,f][0] = W3
        W5[ind,f][1] = W3
        if f in [4]:
            clim = [-0.7, 0.7]
            fig, axes = plt.subplots(1,1,figsize = (10,10))
            im1 = axes.imshow(W3[:,:], clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
            # im2 = axes[1].imshow(W2, aspect = "auto", interpolation = "None")
            # axes.set_xlim([,40])
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(im1, cax=cbar_ax)
        if ind == 0:
            W5IC[f] = W3
        elif ind == 1:           
            W5AC[f] = W3
        # W4IC = W4
    
# print(np.size(np.intersect1d(listOv[0],listOv[3])))
# np.save('PPC_Hist.npy',listOv,allow_pickle = True)
# np.argmax()

# list0n = good_listRu[list0]
# ind= 1
# np.sum((max_peak3[ind,4] > 54) * (max_peak3[ind,4] < 80 ))

# np.sum((max_peak3[ind,4] > 1) * (max_peak3[ind,4] < 40 ))
# np.sum((max_peak3[ind,4] > 80))

# %% calculate nb of neurons encodin

# create list of all neurons that encode at least 1 variable
ind = 0
ax_sz = 10
test = [];
for f in np.arange(ax_sz):
    test = np.concatenate((test,listOv[ind,f]))

test_unique, counts = np.unique(test,return_counts= True)

fig, axes = plt.subplots(1,1,figsize = (10,10))

# sns.histplot(data = counts, stat = "probability")
sns.histplot(data = counts)
# axes.set_xlim([0.5, 4.5])
# axes.set_ylim([0,1])
# axes.hist(counts)
    
    


# %% for each timebin, calculate the number of neurons encoding each TV

cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green']
clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR','Hit_hist','FA_hist']
lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted']

Lic1 =134 # 78
Lg1 =239  #125

ind = 0 # PPCIC or 1 PPCAC
p = 0 # positive or 1 negative

fig, axes = plt.subplots(1,1,figsize = (10,5))
y_all = np.zeros((ax_sz,80))
for f in np.arange(ax_sz):
    list0 = (np.mean(Convdata[f],1) != 0)
        
    Lg = len(good_list2)
    Lic = np.where(good_list2 <194)
    Lic = Lic[0][-1]
    if ind == 0:
        list0[Lic:Lg] = False # PPCIC
    elif ind == 1:           
        list0[0:Lic] = False # PPCAC
        
        # list0ind = np.arange(Lg)
        # list0ind = list0ind[list0]
    list0ind = good_list2[list0]
    W = ndimage.uniform_filter(Convdata[f][list0,:],[0,2], mode = "mirror")
    W = Convdata[f][list0,:]
    SD = np.std(W[:,:])
    test = np.abs(W5[ind,f][p])>0.5*SD
    if ind ==0:        
        y = np.sum(test,0)/Lic1
    elif ind == 1:
        y = np.sum(test,0)/Lg1
        
    y_all[f,:] = y
    if p == 0:
        axes.plot(y,c = cmap[f], linestyle = 'solid', linewidth = 3, label = clabels[f] )
        axes.set_ylim([0,.6])
        axes.legend()
    elif p == 1:
        axes.plot(-y,c = cmap[f], linestyle = 'solid', linewidth = 3 )
        axes.set_ylim([-0.20,0])
        
    
plt.savefig("Fraction of neurons "+ ".svg")
# %% plot positive and negative weights separately.
cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']

pp = 8
maxy = np.zeros((2,10))
for ind in [0,1]:
    fig, axes = plt.subplots(2,2,figsize = (10,10),sharex = "all")
    fig.subplots_adjust(hspace=0)
    for p in [0,1]:
        for f in [pp,pp+1]:
            y1 = ndimage.gaussian_filter1d(np.sum(W5[ind,f][p],0),1)
            y1 = y1/(np.size(W5[ind,f][0],0)+np.size(W5[ind,f][1],0))
            e1 = np.std(W5[ind,f][p],0)/np.sqrt((np.size(W5[ind,f][0],0)+np.size(W5[ind,f][1],0)))
            axes[p,f-pp].plot(xaxis,y1,c = cmap[f],linestyle = 'solid', linewidth = 3)
            axes[p,f-pp].fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)
            # axes[p,f-3].set_xlim([-4,1])
            maxy[p,f] = np.max(np.abs(y1)+np.abs(e1))
    
    # for f in [4,5]:
    #     axes[0,f-4].set_ylim([0, np.max(maxy[:,f])])
    #     axes[1,f-4].set_ylim([-np.max(maxy[:,f]),0])
    
    
    # axes[0,0].set_ylim([0, 0.4])
    # axes[1,0].set_ylim([-0.4,0])
    # axes[0,1].set_ylim([0, 0.3])
    # axes[1,1].set_ylim([-0.3,0])
    
    # plt.savefig("TVencoding"+ str(ind) + "tv" + str(f) + ".svg")
    
# %% plot each weights 
cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green']
lstyles = ['dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted']

pp = 8
maxy = np.zeros((2,10))
fig, axes = plt.subplots(1,1,figsize = (10,5),sharex = "all")
fig.subplots_adjust(hspace=0)
for f in [4]: #np.arange(ax_sz):

    for ind in [0,1]:
        if ind == 0:
            y1 = ndimage.gaussian_filter1d(np.mean(W5IC[f],0),1)
            e1 = np.std(W5IC[f],0)/np.sqrt(np.size(W5IC[f],0))
            # e1 = np.std(W5IC[f],0)/np.sqrt((np.size(W5[0,f][1],0)+np.size(W5[0,f][0],0)))
        elif ind ==1:
            y1 = ndimage.gaussian_filter1d(np.mean(W5AC[f],0),1)
            e1 = np.std(W5AC[f],0)/np.sqrt(np.size(W5AC[f],0))
            # e1 = np.std(W5AC[f],0)/np.sqrt((np.size(W5[1,f][1],0)+np.size(W5[1,f][0],0)))
        axes.plot(xaxis,y1,c = cmap[f],linestyle = lstyles[ind], linewidth = 3)
        axes.fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)
    # ks test
    scat = np.zeros((2,np.size(W5IC[f],1)))
    pcat = np.zeros((2,np.size(W5IC[f],1)))
    for t in np.arange(np.size(W5IC[f],1)):
        s1,p1 = stats.ks_2samp(W5IC[f][:,t], W5AC[f][:,t],'less')
        s2,p2 = stats.ks_2samp(W5AC[f][:,t], W5IC[f][:,t],'less')
        if p1 < 0.05:
            scat[0,t] = True
            pcat[0,t] = p1
        if p2 < 0.05:
            scat[1,t] = True
            pcat[1,t] = p2
    c1 = pcat[0,scat[0,:]>0]
    c2 = pcat[1,scat[1,:]>0]
    axes.scatter(xaxis[scat[0,:]>0],np.ones_like(xaxis[scat[0,:]>0])*0.6,marker='^',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])
    axes.scatter(xaxis[scat[1,:]>0],np.ones_like(xaxis[scat[1,:]>0])*0.6,marker='v',c = np.log10(c2),cmap = 'Greys_r',clim = [-3,0])

        
    axes.set_ylim([-0.1,0.8])
    # for ind in [0,1]:
    #     y1 = ndimage.gaussian_filter1d(np.mean(W5[ind,f][1],0),1)*np.size(W5[ind,f][1],0)/(np.size(W5[0,f][1],0)+np.size(W5[0,f][0],0))
    #     e1 = np.std(W5[ind,f][1],0)/np.sqrt((np.size(W5[0,f][1],0)+np.size(W5[0,f][0],0)))
    #     axes[1].plot(xaxis,y1,c = cmap[f],linestyle = lstyles[ind], linewidth = 3)
    #     axes[1].fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)
    #     axes[1].set_ylim([-0.6,0.15])
    
    
        
        
# %% scatter plot weights, reward and ITI
ind = 0
f = 4

B = np.zeros((2,np.size(W5[ind,f][0],0)))
for n in np.arange(np.size(W5[ind,f][0],0)):
    B[0,n] = np.mean(W5[ind,f][0][n,5:35])
    B[1,n] = np.mean(W5[ind,f][0][n,50:80])
    


fig, ax = plt.subplots(1,1, figsize = (5,5))
ax.scatter(B[0,:],B[1,:])    
ax.set_ylim([-0.9,0.9])
ax.set_xlim([-0.9,0.9])



# %% plot weights positive and negative

for f in [4]: #np.arange(ax_sz):
    fig, axes = plt.subplots(2,1,figsize = (10,10),sharex = "all")
    fig.subplots_adjust(hspace=0)
    for ind in [0,1]:
        if ind == 0:
            y1 = ndimage.gaussian_filter1d(np.mean(W5IC[f],0),1)
            e1 = np.std(W5IC[f],0)/np.size(W5IC[f],0)
        elif ind ==1:
            y1 = ndimage.gaussian_filter1d(np.mean(W5AC[f],0),1)
            e1 = np.std(W5AC[f],0)/np.size(W5AC[f],0)
        axes[ind].plot(xaxis,y1,c = cmap[f],linestyle = 'solid', linewidth = 3)
        axes[ind].fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)

# %% historgra
# %%
fig, axes = plt.subplots(1,1,figsize = (10,5), sharex = True)
fig.tight_layout()
fig.subplots_adjust(hspace=0)

cmap = ['tab:orange','white','tab:blue','white','tab:red','white','black','green','tab:purple','white']

edgec = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']


# edgec = ['tab:blue','tab:red','tab:purple','tab:blue','tab:red','tab:orange','orange']
# b11 = [9,9,6,23,12,18,12]/Lic
# b12 = [7,10,4,13,8,2,1]/Lic

# b21 = [42,17,10,39,30,27,37]/(Lg-Lic)
# b22 = [28,32,27,43,28,2,6]/(Lg-Lic)

# R1 
# b11 = [2,1,23,11,30,4,2,1]/Lic
# b12 = [0,0,1,3,6,0,0,0]/Lic

# b21 = [4,1,33,24,43,7,0,8]/(Lg-Lic)
# b22 = [0,0,9,2,13,2,0,10]/(Lg-Lic)

# R2
# b11 = [5,0,21,4,25,3,3,6]/Lic
# b12 = [0,0,0,1,5,1,1,1]/Lic

# b21 = [2,0,17,17,31,7,0,5]/(Lg-Lic)
# b22 = [0,0,8,3,15,4,0,8]/(Lg-Lic)

# Transition

Lic = 101
Lg = 219
b11 = b_count[0][0,:]/Lic
b12 = b_count[0][1,:]/Lic

b21 = b_count[1][0,:]/Lg
b22 = b_count[1][1,:]/Lg
cmap = ['tab:orange','white','tab:blue','white','tab:red','white','black','green','tab:purple','white']

edgec = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
axes.grid(visible=True,axis = 'y')
# axes[1].grid(visible=True,axis = 'y')
axes.bar(np.arange(8)*3,b11+b12, color = cmap, edgecolor = edgec, alpha = 1, width = 0.5, linewidth = 2, hatch = '/')
# axes[0].bar(np.arange(1)*2+0.7,b21, color = cmap3, alpha = 1, width = 0.5)
# axes[0].bar(np.arange(4)*3+`1.4,b31, color = cmap3, alpha = 0.5, width = 0.5)
axes.set_ylim([0,0.7])

# axes[1].bar(np.arange(8)*2+0.7,-b12, color = cmap, edgecolor = edgec, alpha =1, width = 0.5, linewidth = 2, hatch = '/')
# # axes[1].bar(np.arange(2)*2+0.7,-b22, color = cmap3, alpha = 1, width = 0.5)
# # axes[1].bar(np.arange(4)*3+1.4,-b32, color = cmap3, alpha = 0.5, width = 0.5)
# axes[1].set_ylim([-0.4,0.0])
# # axes[0].set_xlim([-.5,1.5])        


# %% convert and save Convdata
C_data = {}
for f in np.arange(ax_sz):
    C_data[f] = np.zeros((726,80))
    p = 0
    for n in good_list2:
        nn = int(n)
        C_data[f][nn,:] = Convdata[f][p,:]
        p+=1

# fig, axes = plt.subplots(1,1,figsize = (10,10))
# axes.plot(np.mean(C_data[4],0)) 

# %% save listOv
np.save('C_data_RT.npy', C_data,allow_pickle= True)     

# np.save('nlist_R2.npy', listOv,allow_pickle= True)     
# %% analysis with overlapping neurons 

nlist =  {};
nlist[0] = np.load('nlist_R1.npy',allow_pickle = True).item()
nlist[1] = np.load('nlist_RT.npy',allow_pickle = True).item()
nlist[2] = np.load('nlist_R2.npy',allow_pickle = True).item()

CD = {};
CD[0] = np.load('C_data_R1.npy',allow_pickle= True).item()  
CD[1] = np.load('C_data_RT.npy',allow_pickle= True).item()  
CD[2] = np.load('C_data_R2.npy',allow_pickle= True).item()  

# %%
W1 = {};
W2 = {};
W3 = {};
W4 = {};


# %%
ind = 0
f = 4
# int_list = np.intersect1d(nlist[0][ind,f],nlist[2][ind,f])
# int_list2 = np.setxor1d(nlist[2][ind,f], int_list)
# list0 = []
# for n in int_list:
#     nn = int(n)
#     list0 = np.concatenate((list0,np.where(good_list2 == nn)[0]))

# list0 = list0.astype(int)


# W = ndimage.uniform_filter(Convdata[f][list0,:],[0,1], mode = "mirror")
# W4 = {};

good_list = np.arange(np.size(D_ppc,0))
# p = 2
for p in [0,1,2]:
    if ind == 0:
        d_list = good_list < 179
    elif ind == 1:
        d_list = good_list > 180
    
    W = CD[p][f][d_list,:]
        
    
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
            if W[n,max_peak[n]] > 4*SD:
                list1.append(m)
                list3.append(m)
            elif W[n,max_peak[n]] <-4*SD:
                list2.append(m)
                list3.append(m)
            
    max_ind1 = max_ind[list1]  
    max_ind2 = max_ind[list2]     
    max_ind3 = max_ind[list3]
    
    
    W1[p] = W[max_ind1]
    W2[p] = W[max_ind2]    
    W4[p] = np.abs(W[max_ind3])
    W3[p] = np.concatenate((W1[p],W2[p]), axis = 0)
    clim = [-0.7, 0.7]
    fig, axes = plt.subplots(1,1,figsize = (10,10))
    im1 = axes.imshow(W3[p][:,:], clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im1, cax=cbar_ax)


# fig, axes = plt.subplots(1,1,figsize = (10,10))
# axes.plot(np.mean(W4,0)) 

#  re plot weights by list1 and list2

# # max_ind4 = np.concatenate((max_ind1,max_ind2))
# max_ind4 = max_ind3
# for ind in [0,1,2]:
#     W0 = CD[ind][f][d_list,:]
#     W = ndimage.uniform_filter(W0[max_ind4,:],[0,3], mode = "mirror")
#     W = np.abs(W)
#     print(np.sum(np.mean(W,1)>0))

#     clim = [-0.1, 0.7]
#     fig, axes = plt.subplots(1,1,figsize = (10,10))
#     im1 = axes.imshow(W, clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
    
#     fig.subplots_adjust(right=0.85)
#     cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
#     fig.colorbar(im1, cax=cbar_ax)
    
# %% 

fig, axes  = plt.subplots(1,1, figsize = (10,10))
lstyles = ['solid','dotted','dashed']
for p in [0,1,2]:
    y1 = np.mean(W4[p],0)
    y1 = ndimage.gaussian_filter1d(y1,2)
    e1 = np.std(W4[p],0)/np.sqrt(np.size(W4[p],0))
    axes.plot(xaxis,y1,linestyle = lstyles[p])
    axes.fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)



# %% plot positive and negative weights separately.
STD = {}
STD = 0.14626
# STD = 0.11 

fig, axes = plt.subplots(2,1,figsize = (10,10),sharex = "all")
fig.subplots_adjust(hspace=0)
for p in [0,1,2]:
    for pp in [0,1]:
        if pp == 0:
            y1 = np.sum(W1[p],0)
            e1 = np.std(W1[p],0)/np.sqrt(np.size(W1[p],0))
            # for t in np.arange(np.size(y1)):               
            #     s,prob = stats.ttest_1samp(W1[p][:,t],STD,alternative = 'greater')
            #     if prob<0.05:
            #         axes[pp].scatter([(t*window-prestim)*1e-3],[0.30],marker = '*', color = 'black')
        elif pp ==1:
            y1 = np.sum(W2[p],0)
            e1 = np.std(W2[p],0)/np.sqrt(np.size(W2[p],0))
            # for t in np.arange(np.size(y1)):               
            #     s,prob = stats.ttest_1samp(W2[p][:,t],STD,alternative = 'less')
            #     if prob<0.05:
            #         axes[pp].scatter([(t*window-prestim)*1e-3],[0.30],marker = '*', color = 'black')
        y1 = ndimage.gaussian_filter1d(y1,2)/np.size(W4[p],0)
        axes[pp].plot(xaxis,y1,linestyle = lstyles[p])
        axes[pp].fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)

axes[0].set_ylim([-0.05,0.4])
axes[1].set_ylim([-0.4,0.05])

p= 0
test1 = {};
fig, axes = plt.subplots(1,1,figsize = (5,5))
for p in [0,1,2]:
    test1[p] = np.mean(W1[p][:,45:60],1)
    axes.bar([p],np.mean(test1[p]))
    axes.errorbar([p],np.mean(test1[p]),np.std(test1[p])/np.sqrt(np.size(test1[p])),color="black")


axes.set_ylim([0,0.4])


s,p = stats.ks_2samp(test1[1], test1[0])


    
