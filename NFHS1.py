#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pyreadstat
dfw, meta = pyreadstat.read_dta('IAWI22FL.DTA')


# In[2]:


df, meta = pyreadstat.read_dta('IAIR23FL.DTA')


# In[3]:


dfw.head()


# In[3]:


df


# In[4]:


df['n']=df['caseid'].apply(str)
df['n']


# In[5]:


def custom(x):
    x=df['n'].str.rsplit(n=1, expand=True)
    return(x[0])
df['r']=custom('v')
df['r']


# In[6]:


df['n']=df['r'].str.replace(' ', '')
df['n']=df['n'].apply(int)
df['n']


# In[7]:


dfw['n']=dfw['whhid'].apply(str)
dfw['n']


# In[8]:


dfw['n']=dfw['n'].str.replace(' ', '')
dfw['n']=dfw['n'].apply(int)
dfw['n']


# In[9]:


dfw.sort_values("n", inplace = True)
dfw


# In[10]:


dfw.drop_duplicates(subset ="n",
                     keep = False, inplace = True)


# In[13]:


dfw


# In[12]:


b = pd.merge(dfw,df,on ='n', how ='inner')
b


# In[27]:


pyreadstat.write_dta(b, 'meged_wealth_NFHS_1.dta')


# In[4]:


df, meta = pyreadstat.read_dta('meged_wealth_NFHS_1.dta')


# In[5]:


df


# In[5]:


df['B']=0
# creating new column B which considers birth events in last 3 years and women who are currently pregnant
conditions=[
    (df['v008']-df['b3_01']<37),
    (df['b3_01'].isnull()) & (df['v213']==1)
        
]
values= [1,1]

df['B'] = np.select(conditions,values)

# this feature is used to drop the women who are not currently pregnant and have not given birth in last 3 years:
# The rationale is we are only looking into age of first birth or pregnancy 


# In[6]:


df['B'].value_counts()


# In[27]:


df['B']


# In[47]:


df5 = df[['B','b3_01','b3_02','b3_03','b3_04','b3_05','v008','caseid','v213' ]]
rslt_df1 = df5[(df5['B']==1) & (df5['b3_01'].isnull()) ]
rslt_df1


# In[7]:


# the present legal age of marriage is 18, any birth after 18 is recorded as birth at 19 years; 
# We want to look at how is education level and wealth index shows effect on age of first birth 
df= df.drop(index=df[df['v012']<15].index)
df= df.drop(index=df[df['v012']>21].index)
# considering only those who gave birth in last 3 years and are currently pregnant 
df= df.drop(index=df[df['B']==0].index)
# removing outliers that is people who gave birth before age of 15 (TFR is calculated for events between 15-49 years)
df= df.drop(index=df[df['v212']<15].index)
df= df.drop(index=df[df['v106']==9].index)


# In[52]:


df['v106'].isnull().sum(axis = 0)


# In[53]:


plt.figure(figsize=(19,12))
plot=sns.boxplot(x='v106', y='v212', data=df, hue='wlthind5')

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


# In[54]:


df['v212'].isnull().sum(axis = 0)


# In[8]:


df['c212']=df['v212']
df['c212'].fillna(df['v012'],inplace=True)


# In[9]:


df['c212'].isnull().sum(axis = 0)


# In[10]:


df['R']=df['v130']
# creating new column B which considers birth events in last 3 years and women who are currently pregnant
conditions=[
    (df['v130']==0),
    (df['v130']==1),
    (df['v130']==2),
    (df['v130']>2),
    ]
values= [0,1,2,3]

df['R'] = np.select(conditions,values)


# In[11]:


df['n106'] = np.where(df['v106'] == 3, 1, 0)


# In[12]:


df['n212'] = np.where(df['c212'] >= 19, 1, 0)


# In[14]:


df1 = df[['n212', 'n106','wlthind5','R']]


# In[15]:


t=df1.corr()
t


# In[16]:


#checking for missing values
sns.heatmap(df1.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[62]:


from sklearn.model_selection import train_test_split
X = df1[['wlthind5','n106']]
y = df1['n212']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# ** Train and fit a logistic regression model on the training set.**

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[18]:


from sklearn.model_selection import train_test_split
X = df1[['wlthind5','n106','R']]
y = df1['n212']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# ** Train and fit a logistic regression model on the training set.**

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[19]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

plot_confusion_matrix(logmodel, X_test, y_test, cmap='Blues')  
plt.rcParams["figure.figsize"] = (10,5)


# In[64]:


from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
g=df1['n212']
F=df1[['n106','wlthind5']]
F=sm.add_constant(F)

F_train, F_test, g_train, g_test = train_test_split(F,g, test_size=0.7, random_state=42)
logit_model2=sm.Logit(g_train,F_train.astype(float))
result2=logit_model2.fit()
print(result2.summary())


# In[20]:


from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
g=df1['n212']
F=df1[['n106','wlthind5','R']]
F=sm.add_constant(F)

F_train, F_test, g_train, g_test = train_test_split(F,g, test_size=0.7, random_state=42)
logit_model2=sm.Logit(g_train,F_train.astype(float))
result2=logit_model2.fit()
print(result2.summary())


# In[21]:


df['v102'].value_counts()


# In[22]:


df1['n212'].value_counts()


# In[23]:


df1['n106'].value_counts()


# In[ ]:




