#!/usr/bin/env python
# coding: utf-8

# # Fertility_and_Education

# Hypothesis: by delaying women from birth event will reduce the fertility rate
# The trend seen in the boxplot is that once women is into higher education, the age of firt birt inceases drastically atleast by 25%
# 
# The logistic regression model predicts that with completing higher education women is likely to give birth after 19 years
# with 58 percent precission with recall values 0.78 and F1 score as 0.67
# 
# 
# Significance of the study is that, 15th Finacne Commission has allocated 15% of it grants to states on basis of TFR,(desired TFR is 2.1)
# The states higher than TFR 2.1 have the possibility to reduce it by positive measures as keeping women in education
# 
# 

# In[70]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pyreadstat


# # Loading NFHS 3 data taken from IAIR74FL.DTA

# In[2]:


df, meta = pyreadstat.read_dta('pyth_nfhs 3.dta',apply_value_formats=True, formats_as_category=True)
df.head()


# In[ ]:


# knowing the variables used


# In[3]:


## education categories ( 0- no education, 1 -primary, 2 - secondary, 3 - higher  )
df['v106'].unique()


# In[4]:


## wealth categories (1- poorest, 2 - porrer, 3 - middle, 4 - richer, 5 - richest)
df['v190'].unique()


# In[5]:


# number of births in last three years
df['v238'].unique()


# In[71]:


df, meta = pyreadstat.read_dta('pyth_nfhs 3.dta')


# In[4]:


df.head()


# In[6]:


df.info()


# In[7]:


# population age is 15-49 since fertility rates are calculated in between these rates in India
df['v012'].unique()


# In[8]:


df['v012'].nunique()


# # Choosing the population for analysis

# In[ ]:


# since the hypothesis is to delay the age of marriage to 21 which will in return reduce the fertility rate immediately in short term

## On long term there are two major advantages
   # 1. Higher educated women tend to have fewer babies
   # 2. Reduced exposure to chance of pregnancy if women are married after 21 ( policy measures that make sure women are married after the age of 21)


# In[72]:


df['B']=0
# creating new column B which considers birth events in last 3 years and women who are currently pregnant
conditions=[
    (df['v238']==1),
    (df['v238']==2),
    (df['v238']==3),
    (df['v238']==4),
    (df['v238']==5),
    (df['v238']==6),
        (df['v238']==0) & (df['v213']==1),
    
]
values= [1,1,1,1,1,1,1]

df['B'] = np.select(conditions,values)

# this feature is used to drop the women who are not currently pregnant and have not given birth in last 3 years:
# The rationale is we are only looking into age of first birth or pregnancy 


# In[73]:


df['B'].value_counts()


# In[74]:


# the present legal age of marriage is 18, any birth after 18 is recorded as birth at 19 years; 
# We want to look at how is education level and wealth index shows effect on age of first birth 
df= df.drop(index=df[df['v012']<15].index)
df= df.drop(index=df[df['v012']>21].index)
# considering only those who gave birth in last 3 years and are currently pregnant 
df= df.drop(index=df[df['B']==0].index)
df= df.drop(index=df[df['v106']==9].index)
# removing outliers that is people who gave birth before age of 15 (TFR is calculated for events between 15-49 years)
df= df.drop(index=df[df['v212']<15].index)


# In[75]:


df['B'].value_counts()


# # Boxploting to see the trends in fertlity and education along with wealth

# In[7]:


plt.figure(figsize=(19,12))
plot=sns.boxplot(x='v106', y='v212', data=df, hue='v190')

#padded  x-axis label

plt.xlabel("Highest Education Level", fontsize=21, labelpad=20)
plt.ylabel("Age at first Birth", fontsize=21, labelpad=20)
#renaming the axix points
plot.set_xticklabels( ('No Education', 'Primary Education',' Secondary Education',' Higher Education') )
# age=19 line
point1 = [-0.5, 19]
point2 = [4, 19]

x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]
a=plt.plot(x_values, y_values, '-k', linestyle="--", label='age=19')
point1 = [-0.5, 22]
point2 = [4, 22]


#labeling legend with proxy artists 
plt.yticks(fontsize=14)
plt.xticks(fontsize=16)
plt.legend(fontsize=18)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

black_line = mlines.Line2D([], [], color='black',
                          markersize=15, label='age=19')
blue_patch = mpatches.Patch(color='#3274a1', label='poorest')
orange_patch = mpatches.Patch(color='#e1812c', label='poorer')
green_patch = mpatches.Patch(color='#3a923a', label='middle')
red_patch = mpatches.Patch(color='#c03d3e', label='richer')
violet_patch = mpatches.Patch(color='#9372b2', label='richest')
first_legend = plt.legend(handles=[ blue_patch, orange_patch, green_patch, red_patch, violet_patch],fontsize=18)
ax = plt.gca().add_artist(first_legend)
plt.title("NFHS 4 survey 2015-2016:Status of Women(age group 15-21) who gave bith in last three years and currently pregnat; \n Total observation 26859 ", fontsize=18)
plt.legend(handles=a, loc=4,fontsize=18 )
plt.show()


# # cleaning data

# In[57]:


df['v212'].isnull().sum(axis = 0)


# In[102]:


# filling missing values in v212 since we are counting pregnat women and women who have given birth in last 3 years:
#in age group of 15-21


# In[58]:


df['c212']=df['v212']
df['c212'].fillna(df['v012'],inplace=True)


# In[59]:


df['c212'].isnull().sum(axis = 0)


# In[60]:


df['n106'] = np.where(df['v106'] == 3, 1, 0)


# In[61]:


df['n212'] = np.where(df['c212'] >= 19, 1, 0)


# In[62]:


df1 = df[['n212', 'n106','v190',]]


# In[29]:


df1 = df[['n212', 'v106','v190',]]


# In[63]:


df['n106'].value_counts()


# In[64]:


df['n212'] = np.where(df['c212'] >= 19, 1, 0)


# In[23]:


#checking for missing values
sns.heatmap(df1.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[24]:


sns.pairplot(df1, hue='n106')


# In[65]:


t=df1.corr()


# In[66]:


t


# In[27]:


sns.heatmap(t, annot=True, cmap='seismic')


# In[28]:


sns.countplot(x='v190',data=df1)


# # logistic regression

# In[67]:


from sklearn.model_selection import train_test_split
X = df1[['v106','v190']]
y = df1['n212']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ** Train and fit a logistic regression model on the training set.**

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[76]:


from sklearn.model_selection import train_test_split
X = df1[['n106','v190']]
y = df1['n212']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# ** Train and fit a logistic regression model on the training set.**

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[91]:


X_train


# In[ ]:


# further analysis will be done to estimate the TFR if females age of leagal marriage is increaed to 21 using Bongaarts model


# In[40]:



import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

plot_confusion_matrix(logmodel, X_test, y_test, cmap='Blues')  
plt.rcParams["figure.figsize"] = (10,5)


# In[48]:


df['R']=df['v130']
# creating new column B which considers birth events in last 3 years and women who are currently pregnant
conditions=[
    (df['v130']==1),
    (df['v130']==2),
    (df['v130']==3),
    (df['v130']>3),
    ]
values= [1,2,3,4]

df['R'] = np.select(conditions,values)


# In[49]:


df['R'].nunique()


# In[50]:


df['R'].value_counts()


# In[51]:


df1 = df[['n212', 'v106','v190','R']]


# In[52]:


from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
g=df1['n212']
F=df1[['v106','v190', 'R']]
F=sm.add_constant(F)

F_train, F_test, g_train, g_test = train_test_split(F,g, test_size=0.7, random_state=42)
logit_model2=sm.Logit(g_train,F_train.astype(float))
result2=logit_model2.fit()
print(result2.summary())


# In[81]:


df1['n106'].value_counts()


# In[83]:


df1['n212'].value_counts()


# In[84]:


df['v102'].value_counts()


# In[ ]:




