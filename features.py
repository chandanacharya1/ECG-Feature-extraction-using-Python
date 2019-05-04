# -*- coding: utf-8 -*-
"""
Author: CHANDAN ACHARYA.
Date : 1 May 2019.
"""
########################### LIBRARIES #########################################
from __future__ import division
from matplotlib import pyplot as plt
import scipy.io as spio
import numpy as np
import statistics
from scipy.stats import kurtosis
from scipy.stats import skew
import sys
sys.path.append("/home/chandan/python-workspace/")
import BOCPD as ocpd #import bocpd from another file
import cProfile
from functools import partial

########################### DATA PROCESSING ###################################
lib = spio.loadmat('/home/chandan/python-workspace/matlab.mat')
data=lib['v']
rri_list = np.ndarray.tolist(data)
rri = [item for sublist in rri_list for item in sublist] #flat the list

'''get the rolling mean and also plot the data

df=pd.DataFrame(data)
df
RM=df.rolling(window=30).mean().dropna()
fig, ax = plt.subplots(figsize=[18, 16])
ax = fig.add_subplot(2, 1, 1)
ax.plot(df)
ax = fig.add_subplot(2, 1, 2, sharex=ax)
ax.plot(RM)
rm=RM.values #convert df to list'''


'''#hrv values from hrv library

results = time_domain(flat_rri)
print(results)'''

####################### FEATURE DEFINITIONS ###################################
"""TIME DOMAIN"""
#independent function to calculate RMSSD
def calc_rmssd(list):
    diff_nni = np.diff(list)#successive differences
    return np.sqrt(np.mean(diff_nni ** 2))
    
    
 #independent function to calculate AVRR   
def calc_avrr(list):
    return sum(list)/len(list)

 #independent function to calculate SDRR   
def calc_sdrr(list):
    return statistics.stdev(list)

 #independent function to calculate SKEW   
def calc_skew(list):
    return skew(list)

 #independent function to calculate KURT   
def calc_kurt(list):
    return kurtosis(list)

def calc_NNx(list):
    diff_nni = np.diff(list)
    return sum(np.abs(diff_nni) > 50)
    
def calc_pNNx(list):
    length_int = len(list)
    diff_nni = np.diff(list)
    nni_50 = sum(np.abs(diff_nni) > 50)
    return 100 * nni_50 / length_int

"""NON LINEAR DOMAIN"""
 #independent function to calculate SD1
def calc_SD1(list):
    diff_nn_intervals = np.diff(list)
    return np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
 #independent function to calculate SD2
def calc_SD2(list):
    diff_nn_intervals = np.diff(list)
    return np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(\
                   diff_nn_intervals, ddof=1) ** 2)
    
 #independent function to calculate SD1/SD2
def calc_SD1overSD2(list):
      diff_nn_intervals = np.diff(list)
      sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
      sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(\
                    diff_nn_intervals, ddof=1) ** 2)
      ratio_sd2_sd1 = sd2 / sd1
      return ratio_sd2_sd1
    
    
 #independent function to calculate CSI
def calc_CSI(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(\
                  diff_nn_intervals, ddof=1) ** 2)
    L=4 * sd1
    T=4 * sd2
    return L/T
       
 #independent function to calculate CVI
def calc_CVI(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(\
                  diff_nn_intervals, ddof=1) ** 2)
    L=4 * sd1
    T=4 * sd2
    return np.log10(L * T)
 
 #independent function to calculate modified CVI
def calc_modifiedCVI(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(\
                  diff_nn_intervals, ddof=1) ** 2)
    L=4 * sd1
    T=4 * sd2
    return L ** 2 / T

 
#sliding window function
def slidingWindow(sequence,winSize,step):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
 
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence\
                        length.")
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-winSize)/step)+1
    # Do the work
    for i in range(0,int(numOfChunks)*step,step):
        yield sequence[i:i+winSize]
        
####################### FEATURE EXTRACTION ####################################

def feature_extract(list_rri, winSize,step,feature):
    chunklist=list(slidingWindow(list_rri,winSize,step))
    featureList=[]
    if(feature=="RMSSD"):
        for sublist in chunklist:
            featureList.append(calc_rmssd(sublist))
    elif(feature=="AVRR"):
        for sublist in chunklist:
            featureList.append(calc_avrr(sublist))
    elif(feature=="SDRR"):
        for sublist in chunklist:
            featureList.append(calc_sdrr(sublist))
    elif(feature=="SKEW"):
        for sublist in chunklist:
            featureList.append(calc_skew(sublist))
    elif(feature=="KURT"):
        for sublist in chunklist:
            featureList.append(calc_kurt(sublist))
    elif(feature=="NNx"):
        for sublist in chunklist:
            featureList.append(calc_NNx(sublist))
    elif(feature=="pNNx"):
        for sublist in chunklist:
            featureList.append(calc_pNNx(sublist))
    elif(feature=="SD1"):
        for sublist in chunklist:
            featureList.append(calc_SD1(sublist))
    elif(feature=="SD2"):
        for sublist in chunklist:
            featureList.append(calc_SD2(sublist))
    elif(feature=="SD1/SD2"):
        for sublist in chunklist:
            featureList.append(calc_SD1overSD2(sublist))
    elif(feature=="CSI"):
        for sublist in chunklist:
            featureList.append(calc_CSI(sublist))
    elif(feature=="CVI"):
        for sublist in chunklist:
            featureList.append(calc_CVI(sublist))
    elif(feature=="modifiedCVI"):
        for sublist in chunklist:
            featureList.append(calc_modifiedCVI(sublist))
    return featureList    
  
########################### PLOTTING ##########################################
def plot_features(featureList,label):
    plt.title(label)
    plt.plot(featureList)
    plt.show()

###################### CALLING FEATURE METHODS ################################
def browsethroughSeizures(list_rri,winSize,step):
    features=["RMSSD","AVRR","SDRR","SKEW","KURT","NNx","pNNx","SD1","SD2",\
              "SD1/SD2","CSI","CVI","modifiedCVI"]
    for item in features:
        featureList=feature_extract(list_rri,winSize,step,item)
        plot_features(featureList,item)
#################### BAYESIAN CHANGE POINT DETECTION ##########################
####inspired by https://github.com/hildensia/bayesian_changepoint_detection
def bayesianOnFeatures(list_rri,winSize,step):
    features=["RMSSD","AVRR","SDRR","SKEW","KURT","NNx","pNNx","SD1","SD2",\
              "SD1/SD2","CSI","CVI","modifiedCVI"]
    for item in features:
        featureList=feature_extract(list_rri,winSize,step,item)
        featureList=np.asanyarray(featureList)
        Q, P, Pcp = ocpd.offline_changepoint_detection\
        (featureList, partial(ocpd.const_prior,l=(len(featureList)+1))\
         ,ocpd.gaussian_obs_log_likelihood, truncate=-40)
        fig, ax = plt.subplots(figsize=[15, 7])
        ax = fig.add_subplot(2, 1, 1)
        ax.set_title(item)
        ax.plot(featureList[:])
        ax = fig.add_subplot(2, 1, 2, sharex=ax)
        ax.plot(np.exp(Pcp).sum(0))
        
#################### CHANGE POINT DETECTION ##########################

