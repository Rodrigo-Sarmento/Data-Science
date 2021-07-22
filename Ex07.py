#!/usr/bin/env python
# coding: utf-8

# In[3]:


import graphviz
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pydotplus

import io

from scipy import misc

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[6]:


df = pd.read_csv('Desktop\Digo\FACULDADE\IA\data.csv')
df.head(3)


# In[7]:


train, test = train_test_split(df, test_size-round(len(df)*0.3))


# In[8]:


print('Tamanho do set de treino: {},\nTamanho teste: {}'.format(len(train), len(test)))


# In[9]:


df.head()


# In[10]:


tree


# In[11]:


features = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness", "duration_ms"]


# In[12]:


x_train = train[features]
y_train = train['target']

x_test = test[features]
y_test = test['target']


# In[13]:


dct = tree.fit(x_train, y_train)


# In[14]:


def showTree(tree, features, path):
    file=io.StringIO()
    export_graphviz(tree, out_file=file, feature_names=features)
    pydotplus.graph_from_dot_data(file.getvalue()).write_png(path)
    img = misc.imread(path)
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.imshow(img)


# In[15]:


get_ipython().run_line_magic('time', '')
showTree(dct, features, 'minhaprimeiradct.png')


# In[16]:


y_pred = tree.predict(x_test)


# In[17]:


y_pred


# In[18]:


score=accuracy_score(y_test,y_pred)*100


# In[19]:


print('Score = {}'.format(score))


# In[20]:


fromsklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)


# In[21]:


f_ypred=clf.predict(x_test)
score=accuracy_score(y_test,f_ypred)*100
print('Score da decision tree: {}'.format(score))


# In[ ]:




