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

# %% File name and directory

# change fname for filename

fname = 'PPC_GLM_dataset_AllSession_230209.mat'
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
    
    

    if c_ind ==1 or c_ind ==-1: # Rule 1
        r_ind = np.arange(200)
    elif c_ind ==2 or c_ind ==-2: # Rule 2
        r_ind = np.arange(200,np.size(X,0))
        
        
    # Adding previous trial correct vs wrong
    XHist = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
    XHist = XHist[:,None]
    X = np.concatenate((X,XHist),1) # History is added at the end
        
    Y = Y[r_ind,:]
    X = X[r_ind,:]
    X = X[:,[1,3,4]] # removing contingency and correct
        
        
    return X,Y 


# %% glm_per_neuron function code.
# Main functions start here. 

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
    reg = Ridge (alpha = 1e1) #Using a linear regression model with Ridge regression regulator set with alpha = 1

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
    
        

    if c_ind == 1 or c_ind ==2:
        cmap = ['tab:orange','tab:blue']
        clabels = ["action","stim"]        
    elif c_ind == -1 or c_ind == -2:     # c_ind == -3 or c_ind == -4      
        cmap = ['tab:orange','tab:blue','tab:olive']
        clabels = ["action","stim","history"]
        
        
        
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
    # 0 : lick vs no lick
    # 1 : stim 1 vs stim 2
    # 2 : trial History
    if c_ind ==0:
       stim_ind = X[:,2] == 1 
    else:
       stim_ind = X[:,1] == 1     
    

    ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind,:],0),2),
             linewidth = 2.0, color = cmap[1],label = '5 kHz')
    
     
    # ax1.fill_between(x_axis,ndimage.gaussian_filter(np.mean(Y[stim_ind,:],0),2)-stats.sem(Y[stim_ind,:],0),
    #                  ndimage.gaussian_filter(np.mean(Y[stim_ind,:],0),2)+stats.sem(Y[stim_ind,:],0))
    
    ax1.plot(x_axis,ndimage.gaussian_filter(np.mean(Y[np.invert(stim_ind),:],0),2),
             linewidth = 2.0, color = cmap[2],label = '10 kHz')
    
    # ax1.fill_between(x_axis,ndimage.gaussian_filter(np.mean(Y[np.invert(stim_ind),:],0),2)-stats.sem(Y[np.invert(stim_ind),:],0),
    #                  ndimage.gaussian_filter(np.mean(Y[np.invert(stim_ind),:],0),2)+stats.sem(Y[np.invert(stim_ind),:],0))
    
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


# %% Run main GLM code
"""     
Each column of X contains the following information:
    0 : lick vs no lick
    1 : correct vs wrong
    2 : stim 1 vs stim 2
    3 : if exists, would be correct history (previous correct ) 

"""



t_period = 6000
prestim = 500

window = 50 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 10 # number of cv
ca = 0

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [-1,-2]



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
        


# %% Normalized population average of task variable weights

d_list = good_list > 179

d_list3 = good_list <= 179

good_list_sep = good_list[:]

weight_thresh = 1*1e-2



Convdata = {};

for c_ind in c_list:
    if c_ind == -1 or c_ind == -2:
        ax_sz = 3
        cmap3 = ['tab:orange','tab:blue','tab:olive']
    
        
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
    
        # Model_coef = np.abs(Model_coef)/(np.max(np.abs(Model_coef)) + 0.2) # soft normalization value for model_coef
        Model_coef = Model_coef/(np.max(np.abs(Model_coef)) + 0.2) # soft normalization value for model_coef
        
        norm_score = np.mean(Model_score, 1)
        norm_score[norm_score < weight_thresh] = 0
        if np.max(norm_score)>0:
            norm_score = norm_score/np.max(norm_score)
        else:
            norm_score = 0    
        conv = Model_coef*norm_score
        for b_ind in np.arange(np.size(Model_coef, 0)):
            Convdata[c_ind,b_ind][n, :] = conv[b_ind, :]
    
    
    x_axis = np.arange(1, prestim+t_period, window)
    fig, axes = plt.subplots(1,1,figsize = (10,8))
    
    for f in range(ax_sz):
            error = np.std(Convdata[c_ind,f],0)/np.sqrt(np.size(good_list_sep))
            y = ndimage.gaussian_filter(np.mean(Convdata[c_ind,f],0),1)
            axes.plot(x_axis*1e-3-prestim*1e-3,y,c = cmap3[f])
            axes.fill_between(x_axis*1e-3-prestim*1e-3,y-error,y+error,facecolor = cmap3[f],alpha = 0.3)
            axes.set_ylim([-0,0.25])
    
    
    e_lines = np.array([0, 500, 500+1000, 2500+1000])
    e_lines = e_lines+500


# %% PCA

d_list = good_list > 179

d_list3 = good_list <= 179


pca = {};


for c_ind in c_list:
    fig, axs = plt.subplots(ax_sz,6,figsize = (20,20))
    for f in np.arange(ax_sz):
        # pca[f] = SparsePCA(n_components=10,alpha = 0.01)  
        pca[c_ind,f] = PCA(n_components=20) 
        # test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(Convdata[c_ind,f][:,:].T,[1,0]))
        test = pca[c_ind,f].fit_transform(ndimage.gaussian_filter(Convdata[c_ind,f][d_list,:].T,[1,0]))
        
        test = test.T
        for t in range(5):
            axs[f,t].plot(test[t,:],c = cmap3[f])
        axs[f,5].plot(np.cumsum(pca[c_ind,f].explained_variance_ratio_))
        plt.savefig("test.svg", format = 'svg')
    
    

                   

    

#  Subspace overlap analysis
            
n_cv = 100
n_neuron = len(good_list[d_list])


Overlap = {};
Overlap[c_list[0]] = np.zeros((ax_sz,ax_sz,n_cv));
Overlap[c_list[1]] = np.zeros((ax_sz,ax_sz,n_cv));
Overlap_across = np.zeros((ax_sz,ax_sz,n_cv));

O_mean = np.zeros((ax_sz,ax_sz));
O_std = np.zeros((ax_sz,ax_sz));

for f in np.arange(ax_sz):
    for f2 in np.arange(ax_sz):
        for k in np.arange(n_cv):

            for c_ind in c_list:
                n_ind = np.random.choice(n_neuron,int(np.floor(n_neuron*0.9)), replace=False)
                Overlap[c_ind][f,f2,k] = np.max(np.abs(np.dot(pca[c_ind,f].components_[:,n_ind], pca[c_ind,f2].components_[:,n_ind].T)*np.identity(20)))
            Overlap_across[f,f2,k] = np.max(np.abs(np.dot(pca[c_list[0],f].components_[:,n_ind], pca[c_list[1],f2].components_[:,n_ind].T)*np.identity(20)))
        
        O_mean[f,f2] = np.mean(Overlap_across[f,f2,:])
        O_std[f,f2] = np.std(Overlap_across[f,f2,:])



x1 = [.8,1.8,2.8]
y1 = [O_mean[0,0],O_mean[1,1],O_mean[2,2]]
e1 = [O_std[0,0],O_std[1,1],O_std[2,2]]


# %%
fig, ax = plt.subplots()
ax.errorbar(x, y, e, linestyle='None', marker='^')
ax.errorbar(x1, y1, e1, linestyle='None', marker='^')
ax.set_ylim([0,0.6])

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