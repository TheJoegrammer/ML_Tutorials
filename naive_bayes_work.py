
# coding: utf-8

# # Naive Bayes Helper functions
# 
# ## 1: Import useful things for math-related fuctions.

# In[1]:


import math
import numpy as np


# ## 2: NBFeatureGivenLabel function
# Takes a training set features_train and labels_train, the Laplace smoothing constant,<br>
# and tries to estimate the percent of time whether or not each word (V) indicates <br>
# that the article (1,2,...,n) is good or bad.
# 
# ### Variables:
# n: Number of articles.
# V: Number of words.
# 
# 
# ### Inputs
#     # features_train - (n by V) numpy ndarray
#                 #The entry features_train(i,j) is 1 if word j appears in the ith training passage and 0 otherwise.
# 	# labels_train - 1D numpy ndarray of length n
#     # lsc - laplace smoothing constant.
#     
# ### Outputs
#     # P=2 by V array.
#         # Each row corrosponds to the probability that article is good (label=1) or bad (label=0) P(Xi|label) 
#         # Each col corrosponds to a for each word Xi.
#         # Example of element:
#             #(0,3): P(word X_3 is present|label=bad)
#             # For word a and class label b, D(b,a) is the Lap-smoothed estimate.
#         

# In[2]:


def NBFeatureGivenLabel(features_train, labels_train, lsc):
    features_train_shape=features_train.shape
    V=features_train_shape[1]    #V: Number of words.
    n=features_train_shape[0]    #n: Number of articles.
    P = np.ones((2,V))           #P(X_1,X_2,....X_V present|Article is good/bad)
    #Iterate over each word
    for i in range(0,V):
        #Initialize count for times term is found in sad stories.
        found_given_bad=0
        #Initialize count for times term is found in happy stories.
        found_given_good=0
        #Iterate over each story to fill constants
        for j in range(0,n):
            #If the word is found in the story
            if(features_train[j,i]==1):
                #If the word is found in a happy story, update that count.
                #If the word is found in a sad story, update that count.
                if(labels_train[j]==1):
                    found_given_good=found_given_good+1
                else:
                    found_given_bad=found_given_bad+1
                    
        #Find the amount of stories that are happy.
        y_ones=np.sum(labels_train)
        #Find the amount of stories that are sad.
        y_zeros=n-y_ones
        #P(X_i|Y=0)
        to_add_0=(found_given_bad+lsc)/(y_zeros+(lsc*y_zeros))
        #P(X_i|Y=1)
        to_add_1=(found_given_good+lsc)/(y_ones+(lsc*y_ones))    
        P[0,i]=to_add_0
        P[1,i]=to_add_1
    return P


# ## 3: NBLabelPrior function
# Finds the % chance of the story being bad, as a "prior".
# 
# ### Variables:
# n: Number of articles.
# V: Number of words.
# 
# 
# ### Inputs
#     # labels_train - 1D numpy ndarray of length n
#     
# ### Outputs
#     # % chance that story is bad.
# 

# In[3]:


def NBLabelPrior(labels_train):
    #Instantiate # of good stories found.
    p = 0
    #1)Loop through all of n, adding all 0's found to p
    for i in range(0,labels_train.shape[0]):
        if(labels_train[i]==0):
            p=p+1
    return float(p)/float(labels_train.shape[0])


# ## 4: NBClassifier function
# Takes the information you've gathered and tries to guess whether or not the story is bad based on the terms that are present.
# 
# ### Variables:
# n: Number of articles.
# V: Number of words.
# 
# 
# ### Inputs
#     # likelihood: likelihood estimates. If the word is present in the story, tries to determine whether or not the story will be bad and good.
# 	# prior - Scalar detailing the blind guess for an sad story.
#     # features_test - laplace smoothing constant.
#     
# ### Outputs
#     # Truth: 1D array of 1's or 0's, detailing guess of whether or not story is happy or sad.

# In[4]:


def NBClassifier(likelihood, prior, features_test):
    (m,V)=features_test.shape
    guess_truth = np.ones(features_test.shape[0])
    for i in range(0,m):
        is_0=math.log(float(prior),10)
        is_1=math.log(float(1-prior),10)
        for j in range(0,V):
            #Extract from features_test each vocab word (j), and determine if word is there or not there.
            j_word_here=features_test[i,j]
            #If word is there, update the guess based on whether or not term is there
            if (j_word_here):
                is_0=is_0+math.log(float(likelihood[0,j]),10)
                is_1=is_1+math.log(float(likelihood[1,j]),10)
            else:
                is_0=is_0+math.log(float(1-likelihood[0,j]),10)
                is_1=is_1+math.log(float(1-likelihood[1,j]),10)
        guess_truth[i]=(is_1>is_0)
    return guess_truth


# In[5]:


#Below line used to convert jupyter file to .py file
#Comment  out below line to speed up import of file.
get_ipython().system('jupyter nbconvert --to script naive_bayes_work.ipynb')

