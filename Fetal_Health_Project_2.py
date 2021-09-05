#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set_palette('husl')


class DataProcessing:
    
    @staticmethod
    def ShuffleData(X):
        for i in range(len(X)-1, 0, -1):
            tmp = random.randint(0, i+1)
            X.iloc[i], X.iloc[tmp] = X.iloc[tmp], X.iloc[i]
            
        return X
    
    @staticmethod
    def NormalizeData(X):
        df = X.copy()
        for column in df.columns[:-1]:
            #normalizacja z wykorzystaniem metody min-max
            max = df[column].max()
            min = df[column].min()
            df[column] = (df[column] - min) / (max - min)

        df.reset_index(drop=True, inplace=True)

        return df
    
    
    @staticmethod
    def SplitData(X, x, y): # x, y - proporcja dzielenia zbiorow: x + y = 100
        df = X.copy()
        
        rate = x/100
        numberOfRows = newSet.shape[0]
        splitRatio1 = int(numberOfRows * rate) - 1
        X_test = df.loc[0:splitRatio1]
        X_validation = df.loc[splitRatio1+1:numberOfRows-1]

        return X_test, X_validation


# In[2]:


fetals = pd.read_csv('fetal_health.csv')


# In[3]:


sns.violinplot(y='fetal_health',x='severe_decelerations', data=fetals, inner='quartile')


# In[4]:


fetals = fetals.drop('severe_decelerations', axis=1) # below explanation why


# In[5]:


# Fetal health: 1 - Normal 2 - Suspect 3 - Pathological
fetals.head()
fetals.info()


# In[6]:


shuffledSet = DataProcessing.ShuffleData(fetals)
newSet = DataProcessing.NormalizeData(shuffledSet)

newSet.head() #normalized set
shuffledset = DataProcessing.ShuffleData(newSet)
fetalstrain, fetalsvalid = DataProcessing.SplitData(shuffledset, 70, 30)
fetalstrain.head(2126)


# In[7]:


sns.pairplot(fetals, hue='fetal_health', markers='+') #it was maked with severe_decelaration included


# In[8]:




#normalized set would need much more effort to be usable for bayes
#not normalized set:
# shuffledset = DataProcessing.ShuffleData(fetals)
# fetalstrain, fetalsvalid = DataProcessing.SplitData(shuffledset, 70, 30)
# fetalstrain.head(2126)


# In[9]:


class NaiveBayes:
    #srednia
    @staticmethod
    def mean(atr):
        return sum(atr)/len(atr)
    
    #odchylenie standardowe
    def std(attr):
        mean = NaiveBayes.mean(attr)
        sumelem = 0
        for i in attr:
            sumelem += (i-mean)**2
        return math.sqrt(sumelem/len(attr))

    #funkcja guassa
    @staticmethod
    def gauss(x,mean,std):
        if(std == 0):
            std = 0.00000000000000001 #dziala XD
        exponent = np.exp(-(x-mean)**2/(2*std**2))
        return 1/(np.sqrt(2*np.pi*std**2))*exponent
    
    @staticmethod
    def classify(X, sample):
        #seperacja na klasy
        result = {}
        
        for name in fetalstrain['fetal_health'].unique():
            a = fetalstrain.loc[fetalstrain['fetal_health'] == name]
            a.pop('fetal_health')
            gauss_result = 1
            for key,values in a.iteritems():
                average = NaiveBayes.mean(a[key])
                std = NaiveBayes.std(a[key])
                gauss_result*=NaiveBayes.gauss(sample[key], average, std)
            result[name]=gauss_result

        return max(result, key=result.get)


# In[10]:


match = 0

for i in range(0,len(fetalsvalid)):
    sample = fetalsvalid.loc[i+len(fetalstrain)]
    if float(sample['fetal_health']) == NaiveBayes.classify(fetalstrain,sample):
        match+=1

print("Accurancy: ", match/len(fetalsvalid)*100, "%")


# In[ ]:




