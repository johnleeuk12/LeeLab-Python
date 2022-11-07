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
from os.path import join as pjoin

# from matplotlib.colors import BoundaryNorm

# from sklearn.metrics import get_scorer_names, d2_tweedie_score, make_scorer

# %% File name and directory

# change fname for filename

fname = 'CaData4GLM_PPCAC_new.mat'
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


def find_good_data_Ca():
    D_ppc = load_matfile_Ca()
    good_list = np.arange(np.size(D_ppc,0))
    
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
    elif c_ind !=0 and c_ind !=3: # if c_ind is 0 this does not separate rule1 and rule2 in this case, need to add + plot contingency
        if c_ind ==1:
            r_ind = np.arange(200)
        elif c_ind ==2:
            r_ind = np.arange(200,np.size(X,0))
        
        S = S[r_ind,:]
        X = X[r_ind,:]
        X = X[:,1:4]  
        
        
    for w in range(int(t_period/window)):
        y = np.mean(S[:,range(window*w,window*(w+1))],1)*1e3
        Y = np.concatenate((Y,y))
        
    Y = np.reshape(Y,(int(t_period/window),N_trial2)).T
    return X, Y

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
    elif c_ind !=0 and c_ind !=3: # if c_ind is 0 this does not separate rule1 and rule2 in this case, need to add + plot contingency
        if c_ind ==1:
            r_ind = np.arange(200)
        elif c_ind ==2:
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
        X, Y = import_data_w_spikes(n,prestim,t_period,window,c_ind)
    else:
    # if using Ca data
        X, Y = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind)
    
    
    t_period = t_period+prestim
    Yhat = [];
    TT2 = [];
    Intercept = [];
    CI2 = [];
    score = [];
    N_trial2 = np.size(X,0)

    
    # reg = TweedieRegressor(power = 0, alpha = 0)
    reg = Ridge(alpha = 1e1) #Using a linear regression model with Ridge regression regulator set with alpha = 1

    for w in range(int(t_period/window)):
        y = Y[:,w]
        X2 = np.column_stack([np.ones_like(y),X])
        ss= ShuffleSplit(n_splits=k, test_size=0.20, random_state=0)
        cv_results = cross_validate(reg, X, y, cv = ss , 
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
    else:           
        cmap = ['tab:orange', 'tab:green','tab:blue']
        clabels = ["action","correct","stim"]
        
        
        
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



t_period = 4500
prestim = 1000

window = 50 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 10 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [-1]



if ca ==0:
    D_ppc = load_matfile()
    good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    good_list = find_good_data_Ca()


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
            Data[n,c_ind-1] = {"coef" : Model_Theta, "score" : score}   
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

when adding trial history, row 6 is trial history, with best category going up to 5 
"""

def get_best_kernel(b_ind, window, window2, Data, c_ind, ana_period,good_list):
    best_kernel[c_ind] = np.zeros((b_ind,np.size(good_list,0)))


    k = 0;
    for n in good_list2:
        n = int(n)
        mi, bs, coef,beta_weights,mean_score = Model_analysis(n, window, window2, Data,c_ind,ana_period)
        norm_coef = np.abs(coef)
        if bs > weight_thresh:
            best_kernel[c_ind][0,k] = int(mi)
            best_kernel[c_ind][1,k] = int(np.argmax(np.abs(coef)))+1
            best_kernel[c_ind][2,k] = norm_coef[0] 
            best_kernel[c_ind][3,k] = norm_coef[1]
            best_kernel[c_ind][4,k] = norm_coef[2]
            if c_ind ==0:
                best_kernel[c_ind][5,k] = norm_coef[3]
            elif c_ind == -1:
                best_kernel[c_ind][5,k] = norm_coef[3]
                best_kernel[c_ind][6,k] = norm_coef[4]
            # best_kernel[(c_ind-1)*c_ind,k] = int(mi)
            # best_kernel[(c_ind-1)*c_ind+1,k] = int(np.argmax(np.abs(coef)))+1
        else:
            best_kernel[c_ind][2:b_ind,k] = np.ones((1,b_ind-2))*-1    
        k = k+1
        
    return best_kernel

weight_thresh = 2*1e-2


# Here we define the time period for model analysis. 
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
    if c_ind == 0:
        pie_labels = ["Uncategorized", "Contingency", "Action", "Correct","Stimuli"]
        cmap = ['tab:gray','tab:purple', 'tab:orange', 'tab:green','tab:blue']
    elif c_ind == -1:
        pie_labels = ["Uncategorized", "Contingency", "Action", "Correct","Stimuli","history"]
        cmap = ['tab:gray','tab:purple', 'tab:orange', 'tab:green','tab:blue','tab:olive']        
    
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


# rule1_VS_rule2(good_list, best_kernel)

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

if c_ind == 0:
    ax_sz = 4
    
elif c_ind == -1:
    ax_sz = 5

fig, axes = plt.subplots(ax_sz,1,figsize = (5,10))
bins = np.arange(1,20)
x_axis = bins*window2*1e-3/2

cmap3 = ['tab:purple','tab:orange','tab:green','tab:blue','tab:olive']

# ana_period = np.array([0, 4500])

for c_ind in c_list:
    if c_ind == 0:
        b_ind = 6
    elif c_ind == -1:
        b_ind = 7
    else:
        b_ind = 5
        
    best_kernel = get_best_kernel(b_ind, window, window2, Data, c_ind, ana_period,good_list)


for f in range(ax_sz):
    axes[f].scatter(best_kernel[c_ind][0,:],best_kernel[c_ind][f+2,:],c = cmap3[f])
    axes[f].set_ylim([0,1])
    axes[f].set_xticks(bins[1::2], x_axis[1::2])
    axes[f].hlines(y =0.25,
              xmin = bins[0]-1,
              xmax = bins[-1]+1,
              linestyles = 'dashed',
              colors = 'black', 
              linewidth = 2.0)


# %% Calculating number/fraction of neurons with significant coding for each task variable

if c_ind == 0:
    ax_sz = 4
    
elif c_ind == -1:
    ax_sz = 5

Frac= {}
Frac2 = {}
Frac = np.zeros((99,ax_sz))
Frac2 = np.zeros((99,ax_sz))

for n in good_list:
    Model_coef= Data[n,c_ind-1]["coef"]
    Model_score = Data[n,c_ind-1]["score"]
    
    for b_ind in np.arange(np.size(Model_coef,0)):
        SD = np.std(Model_coef[b_ind,0:10])
        coef_bi = np.abs(Model_coef[b_ind,11:] - np.mean(Model_coef[b_ind,0:10])) > 4*SD
        score_bi = np.mean(Model_score[11:,:],1) > weight_thresh
        test = coef_bi*score_bi
        Frac[:,b_ind] = Frac[:,b_ind] + test
        
        SD2 = np.std(Model_coef[b_ind,-10:])
        coef_bi2 = np.abs(Model_coef[b_ind,0:-11] - np.mean(Model_coef[b_ind,-10:])) > 4*SD
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













    