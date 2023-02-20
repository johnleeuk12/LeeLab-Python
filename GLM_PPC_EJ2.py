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

fname = 'CaData4GLM_PPCACandIC_v2.mat'
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
    Y2 = [];
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
    
    if c_ind == -1:                
        # Adding previous trial correct vs wrong
        Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
        Xpre = Xpre[:,None]
        X = np.concatenate((X,Xpre),1)
    elif c_ind ==-2:
        Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
        Xpre = Xpre[:,None]
        X = np.concatenate((X,Xpre),1)
        X = X[:,[0,1,3,4]]
    elif c_ind !=0 and c_ind !=3 and c_ind !=-2: # if c_ind is 0 this does not separate rule1 and rule2 in this case, need to add + plot contingency
        if c_ind ==1 or c_ind ==-3:
            r_ind = np.arange(200)
        elif c_ind ==2 or c_ind ==-4:
            r_ind = np.arange(200,np.size(X,0))
        
        S = S[r_ind,:]
        X = X[r_ind,:]
        X = X[:,1:]
        
    for w in range(int(t_period/window)):
        y = np.mean(S[:,range(window*w,window*(w+1))],1)*1e3
        y2 = np.sum(S[:,range(window*w,window*(w+1))],1)
        Y = np.concatenate((Y,y))
        Y2 = np.concatenate((Y2,y2))
        
    Y = np.reshape(Y,(int(t_period/window),N_trial2)).T
    Y2 = np.reshape(Y2,(int(t_period/window),N_trial2)).T
    return X, Y, Y2

def import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind):
    # D_ppc = load_matfile_Ca()
    N_trial = np.size(D_ppc[n,2],0)
    X = D_ppc[n,2][:,2:6] # task variables

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
    else:
    # only contain conditioning trials    
        Y = Y[201:D_ppc[n,4][0][0]]
        X = X[201:D_ppc[n,4][0][0]]

    
    if c_ind == -1:                
        # Adding previous trial correct vs wrong
        Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
        Xpre = Xpre[:,None]
        X = np.concatenate((X,Xpre),1)
    elif c_ind ==-2:
        Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
        Xpre = Xpre[:,None]
        X = np.concatenate((X,Xpre),1)
        X = X[:,[0,1,3,4]]
    elif c_ind !=0 and c_ind !=3 and c_ind !=-2: # if c_ind is 0 this does not separate rule1 and rule2 in this case, need to add + plot contingency
        if c_ind ==1 or c_ind ==-3:
            r_ind = np.arange(200)
        elif c_ind ==2 or c_ind ==-4:
            r_ind = np.arange(200,np.size(X,0))
        
        
        # Adding previous trial correct vs wrong
        Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
        Xpre = Xpre[:,None]
        X = np.concatenate((X,Xpre),1)
        
        Y = Y[r_ind,:]
        X = X[r_ind,:]
        X = X[:,1:]  
        
        
    return X,Y 
    
# %% Main function for GLM
# %% glm_per_neuron function code

def glm_per_neuron(n,t_period,prestim,window,k,c_ind,ca): 
    # if using spike data
    if ca == 0:
        X, Y, Y2 = import_data_w_spikes(n,prestim,t_period,window,c_ind)
    else:
    # if using Ca data
        X, Y = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
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
        X2 = np.column_stack([np.ones_like(y),X])
        ss= ShuffleSplit(n_splits=k, test_size=0.20, random_state=0)
        y2 = ndimage.gaussian_filter(y,0)
        cv_results = cross_validate(reg, X, y2, cv = ss , 
                                    return_estimator = True, 
                                    scoring = 'explained_variance')
        theta = np.zeros((np.size(X,1),k))
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

    
    TT2 = np.reshape(TT2,(int(t_period/window),np.size(X,1))).T
    CI2 = np.reshape(CI2,(int(t_period/window),np.size(X,1))).T
    score = np.reshape(score,(int(t_period/window),k))
    
    
    
    
    # Figures
    
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))
    
        
    if c_ind == 0:
        cmap = ['tab:purple', 'tab:orange', 'tab:green','tab:blue']
        clabels = ["contin","action","correct","stim"]
    elif c_ind == -1:
        cmap = ['tab:purple', 'tab:orange', 'tab:green','tab:blue','tab:olive']
        clabels = ["contin","action","correct","stim","history"]
    elif c_ind == -2:
        cmap = ['tab:purple', 'tab:orange','tab:blue','tab:olive']
        clabels = ["contin","action","stim","history"]
    elif c_ind == 1 or c_ind ==2:
        cmap = ['tab:orange', 'tab:green','tab:blue']
        clabels = ["action","correct","stim"]        
    else:     # c_ind == -3 or c_ind == -4      
        cmap = ['tab:orange', 'tab:green','tab:blue','tab:olive']
        clabels = ["action","correct","stim","history"]
        
        
        
    x_axis = np.arange(1,t_period,window)
    for c in range(np.size(X,1)):        
        ax2.plot(x_axis,ndimage.gaussian_filter(TT2[c,:],2),linewidth = 2.0, color = cmap[c], label = clabels[c])
        ax2.fill_between(x_axis,(ndimage.gaussian_filter(TT2[c,:],2) - CI2[c,:]),
                        (ndimage.gaussian_filter(TT2[c,:],2 )+ CI2[c,:]), color=cmap[c], alpha = 0.2)
    
    ax2.legend(loc = 'upper right')

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
    if c_ind ==0:
       stim_ind = X[:,3] == 1 
    else:
       stim_ind = X[:,1] == 1     
    

    ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind,:],0),2),
             linewidth = 2.0, color = cmap[1],label = '5 kHz')
    ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[np.invert(stim_ind),:],0),2),
             linewidth = 2.0, color = cmap[2],label = '10 kHz')
    ax1.set_title('Firing rate y')
    ax1.legend(loc = 'upper right')

    
    ax3.plot(x_axis,ndimage.gaussian_filter(np.mean(Yhat[stim_ind,:],0),2),linewidth = 2.0, color = cmap[1])
    ax3.plot(x_axis,ndimage.gaussian_filter(np.mean(Yhat[np.invert(stim_ind),:],0),2),linewidth = 2.0, color = cmap[2]) 
    ax3.set_title('Prediction y_hat')

    ax2.set_title('unit_'+str(n+1))
    ax4.set_title('explained variance')
    ax4.set_ylim(bottom = -2, top = var_top)
    plt.show()
    Model_Theta = TT2
    
    return X, Y, Yhat, Model_Theta, score

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



t_period = 6000
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


Data = {}



# change c_ind and n here. 

for c_ind in c_list:
    t = 0 
    good_list2 = [];
    for n in good_list:
        
        n = int(n)
        # X, Y, Yhat, Model_Theta, score = glm_per_neuron(n, t_period, prestim, window,k,c_ind)
        # Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score} 
        try:
            X, Y, Yhat, Model_Theta, score = glm_per_neuron(n, t_period, prestim, window,k,c_ind,ca)
            Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score, 'Y' : Y}   
            t += 1
            # print(t,"/",len(good_list))
            good_list2 = np.concatenate((good_list2,[n]))
        except KeyboardInterrupt:
            break
        except:
            print("Error, probably not enough trials") 
        


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
    score_var   = np.zeros((1,2*int(Dat_length/bin_size)))
    model_mean  = np.zeros((np.size(Model_Theta,0),2*int(Dat_length/bin_size)))
    
    k = 0;
    for ind in np.arange(0,Dat_length-bin_size/2,int(bin_size/2)):
        ind = int(ind)
        score_mean[0,k] = np.mean(Data[n,c_ind-1]["score"][ind:ind+bin_size,:])
        score_var[0,k]  = np.var(Data[n,c_ind-1]["score"][ind:ind+bin_size,:])
        model_mean[:,k] = np.mean(Model_Theta[:,ind:ind+bin_size],1)
        k = k+1
    
    max_ind = np.argmax(score_mean[0,int(ana_bin[0]):int(ana_bin[1])]) + int(ana_bin[0])
    best_score = score_mean[0,max_ind]
    coef = model_mean[:,max_ind]
    
    
    return max_ind, best_score, coef, model_mean, score_mean

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
        mi, bs, coef,beta_weights,mean_score = Model_analysis(n, window, window2, Data,c_ind,ana_period)
        norm_coef = np.abs(coef)
        # Y_mean = np.mean(Data[n,c_ind-1]["Y"])
        if bs > weight_thresh:
            best_kernel[c_ind][0,k] = int(mi)
            best_kernel[c_ind][1,k] = int(np.argmax(np.abs(coef)))+1
            best_kernel[c_ind][2,k] = norm_coef[0] 
            best_kernel[c_ind][3,k] = norm_coef[1]
            best_kernel[c_ind][4,k] = norm_coef[2]
            if c_ind ==0 or c_ind == -2:  
                best_kernel[c_ind][5,k] = norm_coef[3]
            elif c_ind == -1:
                best_kernel[c_ind][5,k] = norm_coef[3]
                best_kernel[c_ind][6,k] = norm_coef[4]
            # best_kernel[(c_ind-1)*c_ind,k] = int(mi)
            # best_kernel[(c_ind-1)*c_ind+1,k] = int(np.argmax(np.abs(coef)))+1
            elif c_ind == -3 or c_ind == -4:
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
ana_period = np.array([0, 6000])
for c_ind in c_list:
    if c_ind == 0 or c_ind ==-3 or c_ind == -4 or c_ind == -2:
        b_ind = 6
    elif c_ind == -1:
        b_ind = 7
    else:
        b_ind = 5
        
    best_kernel = get_best_kernel(b_ind, window, window2, Data, c_ind, ana_period,good_list)
        

    

# %% plot piechart for all trials

def pie_all_rules(best_kernel): 
    
    # d_list = good_list > 179

    # d_list3 = good_list <= 179
    
    # good_list_sep = np.int_(good_list[d_list])

    
    
    if c_ind == 0:
        pie_labels = ["Uncategorized", "Contingency", "Action", "Correct","Stimuli"]
        cmap = ['tab:gray','tab:purple', 'tab:orange', 'tab:green','tab:blue']
    elif c_ind == -1:
        pie_labels = ["Uncategorized", "Contingency", "Action", "Correct","Stimuli","history"]
        cmap = ['tab:gray','tab:purple', 'tab:orange', 'tab:green','tab:blue','tab:olive']
    elif c_ind == -2:
        pie_labels = ["Uncategorized", "Contingency", "Action","Stimuli","history"]
        cmap = ['tab:gray','tab:purple', 'tab:orange', 'tab:blue','tab:olive']
    
    plt.pie(np.bincount(best_kernel[c_ind][1,:].astype(int)),labels = pie_labels, colors = cmap)
    plt.show() 
    
    print(np.bincount(best_kernel[c_ind][1,:].astype(int)))

pie_all_rules(best_kernel)

# %% analysis comparing rule1 and rule 2
"""    
Pool kernel to see change between Rule 1 and 2
depending on how best kernel changes between rule1 and rule 2 
we define pool kernel that shows whether encoding is acquired, lost or maintained

index: 
    0   :   Uncategorized
    1   :   Acquired (from 0 to encoding)
    2   :   Lost (from encoding to 0)
    3   :   Changed (encoding nature has changed)
    4   :   Maintained, Stim
    5   :   Maintained, Correct
    6   :   Maintained, Action
    7   :   Maintained, Trial history
    

"""


def rule1_VS_rule2(good_list,best_kernel, c_list):

    pool_kernel = np.zeros((1,np.size(good_list,0)))
    
    for n in range(np.size(pool_kernel)):  

        # if best_kernel[2][1,n] != 0 and best_kernel[1][1,n] != best_kernel[2][1,n]: # Acq
        #     pool_kernel[0,n] = 1
        # elif best_kernel[1][1,n] != 0 and best_kernel[2][1,n] != best_kernel[1][1,n]: # Lost
        #     pool_kernel[0,n] = 2
        if best_kernel[c_list[0]][1,n] == 0 and best_kernel[c_list[1]][1,n] != 0: # 
            pool_kernel[0,n] = 1
        elif best_kernel[c_list[1]][1,n] == 0 and best_kernel[c_list[0]][1,n] != 0:
            pool_kernel[0,n] = 2
        elif best_kernel[c_list[0]][1,n] != best_kernel[c_list[1]][1,n] and best_kernel[c_list[0]][1,n] != 0 and best_kernel[c_list[1]][1,n] != 0:
            # if best_kernel[1][1,n] == 1 and best_kernel[2][1,n] ==3:
            #     pool_kernel[0,n] = 4 # Action to stim
            # elif best_kernel[1][1,n] == 3 and best_kernel[2][1,n] ==1:
            #     pool_kernel[0,n] = 5 # Stim to action
            # else:
            pool_kernel[0,n] =3
                 
        elif best_kernel[c_list[0]][1,n] == best_kernel[c_list[1]][1,n] and best_kernel[c_list[1]][1,n] !=0 :
            if best_kernel[c_list[0]][1,n] == 3:
                pool_kernel[0,n] = 4
            elif best_kernel[c_list[0]][1,n] == 1:
                pool_kernel[0,n] = 6
            elif best_kernel[c_list[0]][1,n] == 2:
                pool_kernel[0,n] = 5
            elif best_kernel[c_list[0]][1,n] == 4:
                pool_kernel[0,n] = 7

    # plot pie chart for categories
    
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))
    
    if c_list[0] > 0:
        pie_labels = ["Uncategorized", "Action", "Correct","Stimuli"]
        cmap = ['tab:gray', 'tab:orange', 'tab:green','tab:blue']
        pie_labels2 = ["Uncategorized", "Acq", "Lost","Changed",
                       "Stimuli","Correct","Action",]
        cmap2 = ['tab:gray', 'tab:red', (1,1,0),'tab:brown',
                 'tab:blue','tab:green','tab:orange']
    else:
        pie_labels = ["Uncategorized", "Action", "Correct","Stimuli","History"]
        cmap = ['tab:gray', 'tab:orange', 'tab:green','tab:blue','tab:olive']
        pie_labels2 = ["Uncategorized", "Acq", "Lost","Changed",
                       "Stimuli","Correct","Action","History"]
        cmap2 = ['tab:gray', 'tab:red', (1,1,0),'tab:brown',
                 'tab:blue','tab:green','tab:orange','tab:olive']
    ax1.pie(np.bincount(best_kernel[c_list[0]][1,:].astype(int)),labels = pie_labels, colors = cmap)
    ax2.pie(np.bincount(best_kernel[c_list[1]][1,:].astype(int)),labels = pie_labels, colors = cmap)
    ax1.set_title('Rule1')
    ax2.set_title('Rule2')

    
    #  "Changed,other","Action to Stim","Stim to Action",
    #               'tab:brown',(0.5,1,1),(1,0,1),

    ax3.pie(np.bincount(pool_kernel[0,:].astype(int)),labels = pie_labels2, colors = cmap2)
    plt.show() 
    



# plotting piecharts
# pie_all_rules(best_kernel)


rule1_VS_rule2(good_list, best_kernel,c_list)

# %% Analyzing explained variance across time. This code is mostly for c_ind = 0

binsize = (t_period+prestim)/(window2/2)-1

bins = np.arange(int(binsize))
count_sig = np.zeros((1,int(binsize)))

for c_ind in c_list:
    for n in good_list2:
        n = int(n)
        mi, bs, coef,beta_weights,mean_score = Model_analysis(n, window, window2, Data,c_ind,ana_period)
        for b in bins:
            if mean_score[0,b] > weight_thresh:
                count_sig[0,b] += 1
                
x_axis = bins*window2*1e-3/2 -1  
count_sig = count_sig/np.size(good_list)*1e2              
plt.plot(x_axis,count_sig[0])

fig, ax1 = plt.subplots(1,1,figsize = (5,4))
ax1.bar(x_axis,count_sig[0],
                   width = 0.2,
                   edgecolor ='black')
ax1.set_ylim([0,70])
                
# %% plotting beta weights of all significant neurons 

if c_ind == 0 or c_ind == -2:
    ax_sz = 4
    cmap3 = ['tab:purple','tab:orange','tab:blue','tab:olive']

elif c_ind == -1:
    ax_sz = 5
    cmap3 = ['tab:purple','tab:orange','tab:green','tab:blue','tab:olive']


fig, axes = plt.subplots(ax_sz,1,figsize = (5,10))
bins = np.arange(1,20)
x_axis = bins*window2*1e-3/2


# ana_period = np.array([0, 4500])

for c_ind in c_list:
    if c_ind == 0 or c_ind == -2:
        b_ind = 6
    elif c_ind == -1:
        b_ind = 7
    else:
        b_ind = 5
        
    best_kernel = get_best_kernel(b_ind, window, window2, Data, c_ind, ana_period,good_list)


for f in range(ax_sz):
    axes[f].scatter(best_kernel[c_ind][0,:],best_kernel[c_ind][f+2,:],c = cmap3[f])
    axes[f].set_ylim([0,.5])
    axes[f].set_xticks(bins[1::2], x_axis[1::2])
    # axes[f].hlines(y =0.25,
    #           xmin = bins[0]-1,
    #           xmax = bins[-1]+1,
    #           linestyles = 'dashed',
    #           colors = 'black', 
    #           linewidth = 2.0)


# %% Calculating number/fraction of neurons with significant coding for each task variable

if c_ind == 0:
    ax_sz = 4
    
elif c_ind == -1:
    ax_sz = 5

Frac= {}
Frac2 = {}
Frac = np.zeros((99,ax_sz))
Frac2 = np.zeros((99,ax_sz))
Fthresh = 10;


for n in np.arange(np.size(good_list,0)):
    # n = int(n)
    nn = good_list[n]
    Model_coef= Data[nn,c_ind-1]["coef"]
    Model_score = Data[nn,c_ind-1]["score"]
    if best_kernel[c_ind][1,n] >0:
        for b_ind in np.arange(np.size(Model_coef,0)):
            SD = np.std(Model_coef[b_ind,0:10])
            coef_bi = np.abs(Model_coef[b_ind,11:] - np.mean(Model_coef[b_ind,0:10])) > Fthresh*SD
            score_bi = np.mean(Model_score[11:,:],1) > weight_thresh
            test = coef_bi*score_bi
            Frac[:,b_ind] = Frac[:,b_ind] + test
            
            SD2 = np.std(Model_coef[b_ind,-10:])
            coef_bi2 = np.abs(Model_coef[b_ind,0:-11] - np.mean(Model_coef[b_ind,-10:])) > Fthresh*SD
            score_bi2 = np.mean(Model_score[0:-11,:],1) > weight_thresh
            test2 = coef_bi2*score_bi2
            Frac2[:,b_ind] = Frac2[:,b_ind] + test2
                


        
Frac = Frac/np.size(good_list)
Frac2 = Frac2/np.size(good_list)


fig, axes = plt.subplots(2,1,figsize = (10,12))
cmap3 = ['tab:purple','tab:orange','tab:green','tab:blue','tab:olive']

x_axis = np.arange(1, 4950, window)
e_lines = np.array([0, 500, 500+1000, 2500+1000])
e_lines = e_lines+500




for f in range(ax_sz):
    axes[0].plot(x_axis-500,Frac[:,f]*1e2,c = cmap3[f])

axes[0].vlines(x=e_lines-500,
            ymin=0,
            ymax=50,
            linestyles='dashed',
            colors='black',
            linewidth=2.0)

axes[0].set_ylim([0,np.max(Frac)*1e2+5])

for f in range(ax_sz):
    axes[1].plot(x_axis-1000,Frac2[:,f]*1e2,c = cmap3[f])

axes[1].vlines(x=e_lines-500,
            ymin=0,
            ymax=50,
            linestyles='dashed',
            colors='black',
            linewidth=2.0)

axes[1].set_ylim([0,np.max(Frac2)*1e2+5])

# %% Normalized population average of task variable weights

d_list = good_list > 179

d_list3 = good_list <= 179

cat_list = best_kernel[c_ind][0,:] != 0 # Only neurons that were categorized

good_list_sep = good_list



if c_ind == 0 or c_ind == -2:
    ax_sz = 4
    cmap3 = ['tab:purple','tab:orange','tab:blue','tab:olive']
    
elif c_ind == -1:
    ax_sz = 5
    cmap3 = ['tab:purple','tab:orange','tab:green','tab:blue','tab:olive']

Convdata = {}

for b_ind in np.arange(ax_sz):
    Convdata[b_ind] = np.zeros((np.size(good_list_sep),np.size(score,0)))
    
for n in np.arange(np.size(good_list_sep,0)):
    # n = int(n)
    nn = good_list_sep[n]
    Model_coef = Data[nn, c_ind-1]["coef"]
    Model_score = Data[nn, c_ind-1]["score"]

    Model_coef = np.abs(Model_coef)/(np.max(np.abs(Model_coef)) + 0.1) # soft normalization value for model_coef
    norm_score = np.mean(Model_score, 1)
    norm_score[norm_score < weight_thresh] = 0
    if np.max(norm_score)>0:
        norm_score = norm_score/(np.max(norm_score)+weight_thresh)
    else:
        norm_score = 0    
    conv = Model_coef*norm_score
    for b_ind in np.arange(np.size(Model_coef, 0)):
        Convdata[b_ind][n, :] = conv[b_ind, :]


x_axis = np.arange(1, prestim+t_period, window)
fig, axes = plt.subplots(1,1,figsize = (10,8))

for f in range(ax_sz):
        error = np.std(Convdata[f],0)/np.sqrt(np.size(good_list_sep))
        y = ndimage.gaussian_filter(np.mean(Convdata[f],0),1)
        axes.plot(x_axis*1e-3-prestim*1e-3,y,c = cmap3[f])
        axes.fill_between(x_axis*1e-3-prestim*1e-3,y-error,y+error,facecolor = cmap3[f],alpha = 0.3)
        axes.set_ylim([0,0.25])


e_lines = np.array([0, 500, 500+1000, 2500+1000])
e_lines = e_lines+500

# %% PCA
fig, axs = plt.subplots(ax_sz,6,figsize = (20,20))

d_list = good_list > 179

d_list3 = good_list <= 179


pca = {};
for f in np.arange(ax_sz):
    # pca[f] = SparsePCA(n_components=10,alpha = 0.01)  
    pca[f] = PCA(n_components=20) 
    test = pca[f].fit_transform(ndimage.gaussian_filter(Convdata[f][:,:].T,[1,0]))
    # test = pca[f].fit_transform(ndimage.gaussian_filter(Convdata[f][d_list,:].T,[1,0]))
    
    test = test.T
    for t in range(5):
        axs[f,t].plot(test[t,:],c = cmap3[f])
    axs[f,5].plot(np.cumsum(pca[f].explained_variance_ratio_))
    plt.savefig("test.svg", format = 'svg')


Overlap = np.zeros((ax_sz,ax_sz))

# for f in np.arange(ax_sz): #reference
#     V_cap1 = 1-np.linalg.norm(Convdata[f].T- 
#                              np.dot(np.dot(Convdata[f].T,pca[f].components_.T),pca[f].components_))/np.linalg.norm(Convdata[f].T)

#     for f2 in np.arange(ax_sz): #comparison 
#         V_cap2 = 1-np.linalg.norm(Convdata[f].T- 
#                                  np.dot(np.dot(Convdata[f].T,pca[f2].components_.T),pca[f2].components_))/np.linalg.norm(Convdata[f].T)
#         Overlap[f,f2] = V_cap2/V_cap1
               

for f in np.arange(ax_sz):
    for f2 in np.arange(ax_sz):
        Overlap[f,f2] = np.max(np.dot(pca[f].components_, pca[f2].components_.T))
                        
# fig, ax = plt.subplots(figsize = (10,10))

# ax.imshow(Overlap, cmap='viridis')


# %% Saving files

# np.save('pca_PPCallv2.npy',pca)
# np.save('Data_PPCallv2.npy', Data)
# # pca_IC = np.load('pca_PPCIC.npy',allow_pickle=True).item()
# # ax.set_title('pcolormesh')
# # # set the limits of the plot to the limits of the data
# # ax.clim(0, 1)
# # ax.colorbar()

# %%  plot PC scatter (archive)

# fig, axes = plt.subplots(4,2, figsize = (15,30))

# color_ind = good_list<141

# for f in np.arange(ax_sz):
#     axes[f,0].scatter(pca[f].components_[0,:],pca[f].components_[1,:],s = 15, c= color_ind, cmap = 'bwr')
#     axes[f,0].vlines(0,-0.3,0.3,linestyles = 'dashed')
#     axes[f,0].hlines(0,-0.3,0.3,linestyles = 'dashed')

#     # axes[f,0].axis([-abs(np.max(pca[f].components_[0,:])),abs(np.max(pca[f].components_[0,:])), -abs(np.max(pca[f].components_[1,:])),abs(np.max(pca[f].components_[1,:])) ])
#     axes[f,1].scatter(pca[f].components_[1,:],pca[f].components_[2,:],s = 15, c= color_ind, cmap ='bwr')
#     axes[f,1].vlines(0,-0.3,0.3,linestyles = 'dashed')
#     axes[f,1].hlines(0,-0.3,0.3,linestyles = 'dashed')
    
    
# # test2 = pca[4].components_[0,color_ind]
# # test3 = pca[4].components_[0,good_list >141]

# # fig, ax = plt.subplots(2,1,figsize = (10,10))

# # ax[0].hist(test2, bins = np.arange(-0.4,0.4+0.01,0.01))
# # ax[1].hist(test3, bins = np.arange(-0.4,0.4+0.01,0.01))    
# f = 0



# %% PCA comparison of subspace overlap (Archive)

# d_list = good_list > 179

# d_list3 = good_list <= 179


# cv_ind = 50

# V_cap1_all = np.zeros((5,5,cv_ind))
# V_cap2_all = np.zeros((5,5,cv_ind))
# for cv in np.arange(cv_ind):
    
#     d_list = good_list > 179
    
#     d_list3 = good_list <= 179
#         # 20% shuffle

#     for s in np.arange(np.size(good_list)):
#         if d_list[s] == True:
#             shuffle = np.random.choice(2,1, p = [0.75,0.25])
#             if shuffle == 1:
#                 d_list[s] = False
        
#         if d_list3[s] == True:
#             shuffle = np.random.choice(2,1, p = [0.75,0.25])
#             if shuffle == 1:
#                 d_list3[s] = False


    
#     rand_sample = np.random.randint(2, size=np.size(good_list))
#     rand_sample = np.array(rand_sample, dtype = bool)
    
#     d_list1 = d_list*rand_sample
#     d_list11 = d_list*np.invert(rand_sample)


#     d_list2 = d_list3*rand_sample
#     d_list22 = d_list3*np.invert(rand_sample)
    
#     ref_ind = 1
#     comp_ind = 0
#     V_cap1 = np.zeros((5,5))
#     V_cap11 = np.zeros((5,5))
#     V_cap2 = np.zeros((5,5))
#     V_cap22 = np.zeros((5,5))
    
#     for f  in np.arange(ax_sz): 
#         for ref_ind in np.arange(5):
#             R = ndimage.gaussian_filter(Convdata[f].T,[1,0])
#             V_cap1[f,ref_ind] = 1-np.linalg.norm(R[:,d_list1] - np.dot(np.dot(R[:,d_list1],
#                                                                   pca[f].components_[ref_ind,d_list1].T.reshape(-1,1)),
#                                                                     pca[f].components_[ref_ind,d_list1].T.reshape(1,-1)))/np.linalg.norm(R[:,d_list1])
            
#             V_cap11[f,ref_ind] = 1-np.linalg.norm(R[:,d_list11] - np.dot(np.dot(R[:,d_list11],
#                                                                   pca[f].components_[ref_ind,d_list11].T.reshape(-1,1)),
#                                                                     pca[f].components_[ref_ind,d_list11].T.reshape(1,-1)))/np.linalg.norm(R[:,d_list11])
            
#             V_cap2[f,ref_ind] = 1-np.linalg.norm(R[:,d_list2] - np.dot(np.dot(R[:,d_list2],
#                                                                   pca[f].components_[ref_ind,d_list2].T.reshape(-1,1)),
#                                                                     pca[f].components_[ref_ind,d_list2].T.reshape(1,-1)))/np.linalg.norm(R[:,d_list2])
            
#             V_cap22[f,ref_ind] = 1-np.linalg.norm(R[:,d_list22] - np.dot(np.dot(R[:,d_list22],
#                                                                   pca[f].components_[ref_ind,d_list22].T.reshape(-1,1)),
#                                                                     pca[f].components_[ref_ind,d_list22].T.reshape(1,-1)))/np.linalg.norm(R[:,d_list22])
            
            
            
            
#     V_cap1 = V_cap1 + V_cap11
#     V_cap2 = V_cap2 + V_cap22
      
    
#     V_cap1 = V_cap1.T*(1/np.sum(V_cap1,axis = 1))
#     V_cap1_all[:,:,cv] = V_cap1.T
    
#     V_cap2 = V_cap2.T*(1/np.sum(V_cap2,axis = 1))
#     V_cap2_all[:,:,cv] = V_cap2.T
    
    

# V_cap3 = 1-np.linalg.norm(R-np.dot(np.dot(R,pca[f].components_[ref_ind,:].T.reshape(-1,1)),
#                                     pca[f].components_[ref_ind,:].reshape(1,-1)))/np.linalg.norm(R)




# # PCA subspace overlap, stat tests


# fig, axes = plt.subplots(ax_sz,2,figsize = (5,10))

# x_pos = [1,2]
# for f in np.arange(ax_sz):
    
#     for pc in [0,1]:
#         [s,p] = stats.ttest_ind(V_cap1_all[f,pc,:],V_cap2_all[f,pc,:],equal_var = False)
#         PCmean = [np.mean(V_cap1_all[f,pc,:]),np.mean(V_cap2_all[f,pc,:])]
#         PCerr = [np.std(V_cap1_all[f,pc,:]),np.std(V_cap2_all[f,pc,:])]
#         axes[f,pc].bar(x_pos, PCmean, yerr=PCerr, align='center', alpha=0.5, color = cmap3[f], ecolor='black', capsize=10)
#         axes[f,pc].set_xticks(x_pos)
#         axes[f,pc].set_xticklabels(['PPC_AC', 'PPC_IC'])
#         axes[f,pc].set_ylim([0,1])
        
#         if pc ==1:
#             axes[f,pc].get_yaxis().set_visible(False)
        
#         if p < 0.001:
#             # axes[f,pc].scatter([1.5], [0.75], marker = '*')
#             axes[f,pc].scatter([1.4,1.5,1.6], [0.75,0.75,0.75], marker = '*')
#         if p >0.001 and p<0.01:
#             axes[f,pc].scatter([1.45,1.55], [0.75,0.75], marker = '*')
#         # if p <0.001:
#         #     axes[f,pc].scatter([1.4,1.5,1.6], [0.75,0.75,0.75], marker = '*')
#         # if 0.001< pc<0.01:
#         #     axes[f,pc].scatter([1.5,1.6], [0.75,0.75], marker = '*')
#         # if 0.01 <pc < 0.05:
#         #     axes[f,pc].scatter([1.5], [0.75], marker = '*')



# %% Calculate explained variance by each subspace across time
xtime = np.arange(130)*50*1e-3-prestim*1e-3


d_list1 = good_list > 179
d_list2 = good_list < 179
V_cap1  =np.zeros((ax_sz,130,20))
V_cap2  =np.zeros((ax_sz,130,20))

V_cap1_base = np.zeros((ax_sz,20))
V_cap2_base = np.zeros((ax_sz,20))

fig, axes = plt.subplots(ax_sz,1,figsize = (5,10))
for f  in np.arange(ax_sz): 
    
    
    
    
    R = ndimage.gaussian_filter(Convdata[f].T,[1,0])
    
    for cv in np.arange(20):
        r_shuffle = np.arange(len(good_list))
        np.random.shuffle(r_shuffle)
        R2 = R[:,r_shuffle]
        V_cap1_base[f,cv] = 1-np.linalg.norm(R2 - np.dot(np.dot(R2,pca[f].components_.T),
                                                                pca[f].components_))/np.linalg.norm(R2)
        
                
        d_list1 = good_list > 179
        
        d_list2 = good_list <= 179
            # 20% shuffle
    
        for s in np.arange(np.size(good_list)):
            if d_list1[s] == True:
                shuffle = np.random.choice(2,1, p = [0.75,0.25])
                if shuffle == 1:
                    d_list1[s] = False
            
            if d_list2[s] == True:
                shuffle = np.random.choice(2,1, p = [0.75,0.25])
                if shuffle == 1:
                    d_list2[s] = False
     
        # d_list1 = good_list >0
        # V_cap2_base[f,cv] = 1-np.linalg.norm(R[:,d_list2] - np.dot(np.dot(R[:,d_list2],
        #                                                       pca[f].components_[:,d_list2].T),
        #                                                         pca[f].components_[:,d_list2]))/np.linalg.norm(R[:,d_list2])
        
        for t in np.arange(130):    
            V_cap1[f,t,cv] = 1-np.linalg.norm(R[t,d_list1] - np.dot(np.dot(R[t,d_list1],
                                                                  pca[f].components_[:,d_list1].T),
                                                                    pca[f].components_[:,d_list1]))/np.linalg.norm(R[t,d_list1])
            V_cap2[f,t,cv] = 1-np.linalg.norm(R[t,d_list2] - np.dot(np.dot(R[t,d_list2],
                                                                  pca[f].components_[:,d_list2].T),
                                                                    pca[f].components_[:,d_list2]))/np.linalg.norm(R[t,d_list2])
    
        
    
    
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

# ref_ind = 2
# comp_ind = 0
# angle1 = np.dot(pca_all[4].components_[ref_ind,d_list], pca_AC[4].components_[comp_ind,:])
# angle2 = np.dot(pca_all[4].components_[ref_ind,d_list2], pca_IC[4].components_[comp_ind,:])

A = {}
for f in np.arange(ax_sz):
    A[f] = np.zeros((2,4))
    for ref_ind in [0,1]:
        for comp_ind in [0,1]:
            A[f][ref_ind,comp_ind] = np.dot(pca_all[f].components_[ref_ind,d_list], pca_AC[f].components_[comp_ind,:])
            A[f][ref_ind,comp_ind+2] = np.dot(pca_all[f].components_[ref_ind,d_list2], pca_IC[f].components_[comp_ind,:])
    A[f] = np.abs(A[f])
    
    
    
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
            plt.savefig('images/img' + str(n).zfill(3) + '.png',
                        bbox_inches='tight')
    
# %%

plt.close()
draw_traj(traj,3)
import IPython.display as IPdisplay
import glob
from PIL import Image as PIL_Image
import imageio

images = [PIL_Image.open(image) for image in glob.glob('images/*.png')]
file_path_name = 'images/trajectory_history.gif'
imageio.mimsave(file_path_name, images)

# writeGif(file_path_name, images, duration=0.1)
# IPdisplay.Image(url=file_path_name)
# %%

fig, axes = plt.subplots(ax_sz,2,figsize = (5,10))
traj = {}
xtime = np.arange(130)*5*1e-3-1e3
for f in np.arange(ax_sz):
    R = ndimage.gaussian_filter(Convdata[f].T,[1,0])
    
    traj[f] = {}
    # traj[f][0] = pca[f].fit_transform(R)
    traj[f][0] = np.dot(R,pca[f].components_.T)                                   
    traj[f][1] = np.dot(R[:,d_list1], pca[f].components_[:,d_list1].T) #*(len(good_list)/np.sum(d_list1))
    traj[f][2] = np.dot(R[:,d_list3], pca[f].components_[:,d_list3].T) #*(len(good_list)/np.sum(d_list3))
    traj[f][3] = traj[f][1] + traj[f][2]

    draw_traj(traj,f,0)
    # distance = {}
    # distance[0] = np.linalg.norm(traj[f][0][:,0:3]-traj[f][1][:,0:3],axis = 1)
    # distance[1] = np.linalg.norm(traj[f][0][:,0:3]-traj[f][2][:,0:3],axis = 1)
    
    
    # axes[f,0].plot(xtime,distance[0], linestyle = 'solid')
    # axes[f,0].plot(xtime,distance[1], linestyle = 'dashed')
 
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













    