#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from collections import Counter
import pandas as pd
import random
data=pd.read_csv("C:\\Users\\MNV\\Downloads\\breast-cancer-wisconsin.txt")


# In[2]:


data.replace("?",-99999,inplace=True)
data.drop(["id"],axis=1,inplace=True)
fulldata=data.astype("float").values.tolist()


# In[3]:


random.shuffle(fulldata)
test_size=0.2
train_set={2:[],4:[]}
test_set={2:[],4:[]}
train_data=fulldata[:-int(test_size*len(fulldata))]
test_data=fulldata[-int(test_size*len(fulldata)):]


# In[4]:


def k_nearest_neighbors(data,predict,k):
    if(len(data)>=k):
        print(" k is low")
    distance=[]
    for group in data:
        for feature in data[group]:
            eclidean_distance=np.linalg.norm(np.array(feature)-np.array(predict))
            distance.append([eclidean_distance,group])
    votes=[i[1] for i in sorted(distance)[:k]]
    print(votes)
    vote_result=Counter(votes).most_common(1)[0][0]
    return vote_result
        


# In[5]:


for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct=0
total=0
for group in test_set:
    for j in test_set[group]:
        vote=k_nearest_neighbors(train_set,j,k=5)
        if(group==vote):
            correct+=1
        total+=1


# In[9]:


print(correct/total)


# In[ ]:




