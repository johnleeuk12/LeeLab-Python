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
"""

# import packages 

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
from scipy import stats
from sklearn.linear_model import TweedieRegressor, Ridge, ElasticNet
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
# from matplotlib.colors import BoundaryNorm

# from sklearn.metrics import get_scorer_names, d2_tweedie_score, make_scorer

# %% 
# Helper functions for loading data and selecting good units with FR >1spks/s

# change fname for filename

# fname = 'GLM_dataset_AC_new.mat'
fname = 'GLM_dataset_220824_new.mat'

np.seterr(divide = 'ignore') 
def load_matfile(dataname = fname):
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_dataset']
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




# %% Main function for GLM
"""
This function runs GLM for each neuron n
    INPUT:
    t_period    :   The total trial time (in ms)
    window      :   t_period is divided into bins of size = window (in ms)
    k           :   number of cross-validation permutations
    c_ind       :   index used to separate different analyses
        -1 : all trials except conditioning. Adds the trial correct history factor
        0  : all trials except conditioning, does not factor in trial history
        1  : rule 1
        2  : rule 2
        3  : conditioning trials only
        
"""

def glm_per_neuron(n,t_period,window,k,c_ind): 
    D_ppc = load_matfile()
    S_all = np.zeros((1,max(D_ppc[n,2][:,0])+t_period+100))
    N_trial = np.size(D_ppc[n,2],0)
    
    
    prestim = 1000
    
    
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
    Yhat = [];
    TT2 = [];
    Intercept = [];
    CI2 = [];
    score = [];
    S = np.concatenate((S_pre,S),1)
    t_period = t_period+prestim
    
    
    # remove conditioning trials 
    
    S = np.concatenate((S[0:200,:],S[D_ppc[n,5][0][0]:,:]),0)
    X = np.concatenate((X[0:200,:],X[D_ppc[n,5][0][0]:,:]),0)
    
    # only contain conditioningt trials
    
    # S = S[201:D_ppc[n,5][0][0]]
    # X = X[201:D_ppc[n,5][0][0]]

    # select analysis and model parameters with c_ind    
    
    if c_ind == -1:                
        # Adding previous trial correct vs wrong
        Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
        Xpre = Xpre[:,None]
        X = np.concatenate((X,Xpre),1)
    elif c_ind !=0: # if c_ind is 0 this does not separate rule1 and rule2 in this case, need to add + plot contingency
        if c_ind ==1:
            r_ind = np.arange(200)
        elif c_ind ==2:
            r_ind = np.arange(200,np.size(X,0))
        
        S = S[r_ind,:]
        X = X[r_ind,:]
        X = X[:,1:4]

    
    N_trial2 = np.size(S,0)
    
    # 0 : contingency 
    # 1 : lick vs no lick
    # 2 : correct vs wrong
    # 3 : stim 1 vs stim 2
    # 4 : if exists, would be correct history (previous correct )
    
    # reg = TweedieRegressor(power = 0, alpha = 0)
    reg = Ridge(alpha = 1e1) #Using a linear regression model with Ridge regression regulator set with alpha = 1

    for w in range(int(t_period/window)):
        y = np.mean(S[:,range(window*w,window*(w+1))],1)*1e3
        X2 = np.column_stack([np.ones_like(y),X])
        ss= ShuffleSplit(n_splits=k, test_size=0.20, random_state=0)
        cv_results = cross_validate(reg, X, y, cv = ss , 
                                    return_estimator = True, 
                                    scoring = 'explained_variance')
        # cv_results2 = cross_val_score(reg, X,y ,cv = 5,scoring = 'r2')
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

        Y = np.concatenate((Y,y))
        Yhat = np.concatenate((Yhat,yhat))
        
    Y = np.reshape(Y,(int(t_period/window),N_trial2)).T
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
    else:           
        cmap = ['tab:orange', 'tab:green','tab:blue']
        clabels = ["action","correct","stim"]
        
        
        
    x_axis = np.arange(1,t_period,window)
    for c in range(np.size(X,1)):        
        ax2.plot(x_axis,ndimage.gaussian_filter(TT2[c,:],3),linewidth = 2.0, color = cmap[c], label = clabels[c])
        ax2.fill_between(x_axis,(ndimage.gaussian_filter(TT2[c,:],3) - CI2[c,:]),
                        (ndimage.gaussian_filter(TT2[c,:],3 )+ CI2[c,:]), color=cmap[c], alpha = 0.2)
    
    ax2.legend(loc = 'upper right')

    e_lines = np.array([0,500,500+int(D_ppc[n,3]),2500+int(D_ppc[n,3])])
    e_lines = e_lines+prestim

    
    ax2.vlines(x =e_lines, 
              ymin = np.amin(ndimage.gaussian_filter(TT2,sigma = [0,3])), 
              ymax = np.amax(ndimage.gaussian_filter(TT2,sigma = [0,3])),
              linestyles = 'dashed',
              colors = 'black', 
              linewidth = 2.0)
    
    ax4.plot(x_axis,ndimage.gaussian_filter(np.mean(score,1)*1e2,1))
    
    
    # Plotting firing rates for one condition VS the other
    # 0 : contingency 
    # 1 : lick vs no lick
    # 2 : correct vs wrong
    # 3 : stim 1 vs stim 2
    if c_ind ==0:
       stim_ind = X[:,3] == 1 
    else:
       stim_ind = X[:,2] == 1     
    

    ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind,:],0),2),
             linewidth = 2.0, color = cmap[1],label = '5 kHz')
    ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[np.invert(stim_ind),:],0),2),
             linewidth = 2.0, color = cmap[2],label = '10 kHz')
    ax1.set_title('Firing rate y')
    ax1.legend(loc = 'upper right')

    
    ax3.plot(x_axis,ndimage.gaussian_filter(np.mean(Yhat[stim_ind,:],0),2),linewidth = 2.0, color = cmap[1])
    ax3.plot(x_axis,ndimage.gaussian_filter(np.mean(Yhat[np.invert(stim_ind),:],0),2),linewidth = 2.0, color = cmap[2]) 
    ax3.set_title('Prediction y_hat')

    ax2.set_title('unit_'+str(n))
    ax4.set_title('explained variance')
    plt.show()
    Model_Theta = TT2
    
    return X, Y, Yhat, Model_Theta, score
        

t_period = 4500
window = 50 # averaging firing rates with this window 
window2 = 500
k = 100 # number of cv
# n = 109

good_list = find_good_data()

Data = {}
c_list = [0]



# change c_ind and n here. 

for c_ind in c_list:    
    for n in good_list:
        n = int(n)
        try:
            X, Y, Yhat, Model_Theta, score = glm_per_neuron(n, t_period, window,k,c_ind)
            Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score}   
        except:
            print("Error, probably not enough trials") 
        finally:
            print("")

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

when adding trial history, row 6 is trial history, with best category going up to 5 
"""
# change c_ind range according to 

def get_best_kernel(b_ind, window, window2, Data, c_ind, ana_period,good_list):
    best_kernel[c_ind] = np.zeros((b_ind,np.size(good_list,0)))


    k = 0;
    for n in good_list:
        n = int(n)
        mi, bs, coef,beta_weights,mean_score = Model_analysis(n, window, window2, Data,c_ind,ana_period)
        norm_coef = np.abs(coef)
        if bs > weight_thresh and bs<60*1e-2:
            best_kernel[c_ind][0,k] = int(mi)
            best_kernel[c_ind][1,k] = int(np.argmax(np.abs(coef)))+1
            best_kernel[c_ind][2,k] = norm_coef[0] 
            best_kernel[c_ind][3,k] = norm_coef[1]
            best_kernel[c_ind][4,k] = norm_coef[2]
            if c_ind ==0:
                best_kernel[c_ind][5,k] = norm_coef[3]
            elif c_ind == -1:
                best_kernel[c_ind][6,k] = norm_coef[4]
            # best_kernel[(c_ind-1)*c_ind,k] = int(mi)
            # best_kernel[(c_ind-1)*c_ind+1,k] = int(np.argmax(np.abs(coef)))+1
        else:
            best_kernel[c_ind][2:b_ind,k] = np.ones((1,b_ind-2))*-1    
        k = k+1
        
    return best_kernel

weight_thresh = 0.5*1e-2
# ana_period = np.array([1000, 1500]) # (Stimulus presentation period)
# ana_period = np.array([1500, 2500])
# ana_period = np.array([2500, 4500])
ana_period = np.array([0, 4500])
for c_ind in c_list:
    if c_ind == 0:
        b_ind = 6
    elif c_ind == -1:
        b_ind = 7
    else:
        b_ind = 5
        
    best_kernel = get_best_kernel(b_ind, window, window2, Data, c_ind, ana_period,good_list)
        

    

# %% plot piechart for all trials

def pie_all_rules(best_kernel): 
    
    pie_labels = ["Uncategorized", "Contingency", "Action", "Correct","Stimuli"]
    cmap = ['tab:gray','tab:purple', 'tab:orange', 'tab:green','tab:blue']
    # pie_labels = ["Uncategorized", "Action", "Correct","Stimuli"]
    # cmap = ['tab:gray', 'tab:orange', 'tab:green','tab:blue']
    plt.pie(np.bincount(best_kernel[0][1,:].astype(int)),labels = pie_labels, colors = cmap)
    plt.show() 
    
    print(np.bincount(best_kernel[0][1,:].astype(int)))

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
    

"""
def rule1_VS_rule2(good_list,best_kernel):

    pool_kernel = np.zeros((1,np.size(good_list,0)))
    
    for n in range(np.size(pool_kernel)):  

        # if best_kernel[2][1,n] != 0 and best_kernel[1][1,n] != best_kernel[2][1,n]: # Acq
        #     pool_kernel[0,n] = 1
        # elif best_kernel[1][1,n] != 0 and best_kernel[2][1,n] != best_kernel[1][1,n]: # Lost
        #     pool_kernel[0,n] = 2
        if best_kernel[1][1,n] == 0 and best_kernel[2][1,n] != 0:
            pool_kernel[0,n] = 1
        elif best_kernel[2][1,n] == 0 and best_kernel[1][1,n] != 0:
            pool_kernel[0,n] = 2
        elif best_kernel[1][1,n] != best_kernel[2][1,n] and best_kernel[1][1,n] != 0 and best_kernel[2][1,n] != 0:
            # if best_kernel[1][1,n] == 1 and best_kernel[2][1,n] ==3:
            #     pool_kernel[0,n] = 4 # Action to stim
            # elif best_kernel[1][1,n] == 3 and best_kernel[2][1,n] ==1:
            #     pool_kernel[0,n] = 5 # Stim to action
            # else:
            pool_kernel[0,n] =3
                 
        elif best_kernel[1][1,n] == best_kernel[2][1,n] and best_kernel[2][1,n] !=0 :
            if best_kernel[1][1,n] == 3:
                pool_kernel[0,n] = 4
            elif best_kernel[1][1,n] == 1:
                pool_kernel[0,n] = 6
            else:
                pool_kernel[0,n] = 5

    # plot pie chart for categories
    
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))
    
    pie_labels = ["Uncategorized", "Action", "Correct","Stimuli"]
    cmap = ['tab:gray', 'tab:orange', 'tab:green','tab:blue']
    ax1.pie(np.bincount(best_kernel[1][1,:].astype(int)),labels = pie_labels, colors = cmap)
    ax2.pie(np.bincount(best_kernel[2][1,:].astype(int)),labels = pie_labels, colors = cmap)
    ax1.set_title('Rule1')
    ax2.set_title('Rule2')
    pie_labels2 = ["Uncategorized", "Acq", "Lost","Changed",
                   "Stimuli","Correct","Action",]
    cmap2 = ['tab:gray', 'tab:red', (1,1,0),'tab:brown',
             'tab:blue','tab:green','tab:orange']
    
    #  "Changed,other","Action to Stim","Stim to Action",
    #               'tab:brown',(0.5,1,1),(1,0,1),

    ax3.pie(np.bincount(pool_kernel[0,:].astype(int)),labels = pie_labels2, colors = cmap2)
    plt.show() 
    



# plotting piecharts
# pie_all_rules(best_kernel)
rule1_VS_rule2(good_list, best_kernel)

# %% Analyzing weights across time. This code is mostly for c_ind = 0

bins = np.arange(21)
count_sig = np.zeros((1,21))

for c_ind in c_list:
    for n in good_list:
        n = int(n)
        mi, bs, coef,beta_weights,mean_score = Model_analysis(n, window, window2, Data,c_ind,ana_period)
        for b in bins:
            if mean_score[0,b] > weight_thresh:
                count_sig[0,b] += 1
                
x_axis = bins*0.250 -1  
count_sig = count_sig/np.size(good_list)*1e2              
plt.plot(x_axis,count_sig[0])

fig, ax1 = plt.subplots(1,1,figsize = (5,4))
ax1.bar(x_axis,count_sig[0],
                   width = 0.2,
                   edgecolor ='black')
ax1.set_ylim([0,70])
                
# %% plotting beta weights of all significant neurons 

fig, axes = plt.subplots(4,1,figsize = (5,10))
bins = np.arange(1,20)
cmap3 = ['tab:purple','tab:orange','tab:green','tab:blue']
x_axis = bins*0.250

ana_period = np.array([0, 4500])

for c_ind in [0]:
    if c_ind == 0:
        b_ind = 6
    elif c_ind == -1:
        b_ind = 7
    else:
        b_ind = 5
        
    best_kernel = get_best_kernel(b_ind, window, window2, Data, c_ind, ana_period,good_list)


for f in range(4):
    axes[f].scatter(best_kernel[0][0,:],best_kernel[0][f+2,:],c = cmap3[f])
    axes[f].set_ylim([0,1])
    axes[f].set_xticks(bins[1::2], x_axis[1::2])
    axes[f].hlines(y =0.25,
              xmin = bins[0]-1,
              xmax = bins[-1]+1,
              linestyles = 'dashed',
              colors = 'black', 
              linewidth = 2.0)


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













    