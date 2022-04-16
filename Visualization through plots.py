#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pyreadstat
from sklearn import preprocessing
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
xls = pd.ExcelFile('B12.xlsx')
df1 = pd.read_excel(xls, 'Bihar')
df1 = df1.iloc[0: , :]
df1.head()


# In[ ]:





# In[2]:


df2 = pd.read_excel(xls, 'Punjab')
df2 = df2.iloc[0: , :]
df2.head()


# In[3]:


df3 = pd.read_excel(xls, 'Himachal pradesh')
df3 = df3.iloc[0: , :]
df3.head()


# In[4]:


df4 = pd.read_excel(xls, 'Gujarat')
df4 = df4.iloc[0: , :]
df4.head()


# In[5]:


df5 = pd.read_excel(xls, 'Kerala')
df5 = df5.iloc[0: , :]
df5.head()


# In[6]:


df6 = pd.read_excel(xls, 'Sikkim')
df6 = df6.iloc[0: , :]
df6.head()


# In[ ]:





# In[4]:


fig = plt.figure()
ax=fig.add_axes([0,0,1.8,1.8])


ax.plot(df1.ageg, df1['Actual'],lw=3)
ax.plot(df1.ageg, df1['rural actual 2'],lw=3)
ax.plot(df1.ageg, df1['urban actual'],lw=3)


ax.set_title('True predction of higher education and age of first birth- recall values  ', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend(['first birth below 19 years', 'first birth above 19 years'],fontsize=18, loc=4)


# In[7]:


df1['ageg 2'].dtype


# In[16]:


fig, axes = plt.subplots(nrows=3, ncols=2,figsize=(12,18))
fig = plt.figure()


axes[0,0].set_xlim([14, 50])
axes[0,0].set_ylim([0, .34])
axes[0,0].plot(df1.ageg, df1['Actual'],lw=3)
axes[0,0].plot(df1.ageg, df1['rural actual 2'],lw=3)
axes[0,0].plot(df1.ageg, df1['urban actual'],lw=3)

#punjab


axes[0,1].plot(df2.ageg, df2['Actual'],lw=3)
axes[0,1].set_xlim([14, 50])
axes[0,1].set_ylim([0, .34])
axes[0,1].plot(df2.ageg, df2['rural actual'],lw=3)
axes[0,1].plot(df2.ageg, df2['urban actual'],lw=3)

axes[0].set_title('True predction of higher education and age of first birth- recall values  ', fontsize=18)

axes[0].plt.legend(['first birth below 19 years', 'first birth above 19 years'],fontsize=18, loc=4)


# In[20]:


fig, axes = plt.subplots(nrows=3, ncols=2,figsize=(12,18))
fig = plt.figure()

#bihar

axes[0,0].set_xlim([14, 50])
axes[0,0].set_ylim([0, .34])
axes[0,0].plot(df1.ageg, df1['Actual'],lw=3)
axes[0,0].plot(df1.ageg, df1['rural actual'],lw=3)
axes[0,0].plot(df1.ageg, df1['urban actual'],lw=3)

#punjab

axes[0,1].set_xlim([14, 50])
axes[0,1].set_ylim([0, .34])
axes[0,1].plot(df2.ageg, df2['Actual'],lw=3)
axes[0,1].plot(df2.ageg, df2['rural actual'],lw=3)
axes[0,1].plot(df2.ageg, df2['urban actual'],lw=3)

#Himachal pradesh

axes[1,0].set_xlim([14, 50])
axes[1,0].set_ylim([0, .34])
axes[1,0].plot(df3.ageg, df3['Actual'],lw=3)
axes[1,0].plot(df3.ageg, df3['rural actual'],lw=3)
axes[1,0].plot(df3.ageg, df3['urban actual'],lw=3)

#Gujarat

axes[1,1].set_xlim([14, 50])
axes[1,1].set_ylim([0, .34])
axes[1,1].plot(df4.ageg, df4['Actual'],lw=3)
axes[1,1].plot(df4.ageg, df4['rural actual'],lw=3)
axes[1,1].plot(df4.ageg, df4['urban actual'],lw=3)

#Kerala

axes[2,0].set_xlim([14, 50])
axes[2,0].set_ylim([0, .34])
axes[2,0].plot(df5.ageg, df5['Actual'],lw=3)
axes[2,0].plot(df5.ageg, df5['rural actual'],lw=3)
axes[2,0].plot(df5.ageg, df5['urban actual'],lw=3)

#Sikkim

axes[2,1].set_xlim([14, 50])
axes[2,1].set_ylim([0, .34])
axes[2,1].plot(df6.ageg, df6['Actual'],lw=3)
axes[2,1].plot(df6.ageg, df6['rural actual'],lw=3)
axes[2,1].plot(df6.ageg, df6['urban actual'],lw=3)



# In[43]:


fig, axes = plt.subplots(nrows=4, ncols=3,figsize=(20,20))
fig = plt.figure()

axes[0,0].set_title('Bihar age specific fertility rates  ', fontsize=18)
axes[0,0].set_ylabel('Observed', fontsize=18)
axes[0,0].set_xlim([14, 50])
axes[0,0].set_ylim([0, .34])
axes[0,0].plot(df1.ageg, df1['Actual'],lw=1.25)
axes[0,0].plot(df1.ageg, df1['rural actual'],lw=1.25)
axes[0,0].plot(df1.ageg, df1['urban actual'],lw=1.25)

#punjab
axes[0,1].set_title('Punjab age specific fertility rates  ', fontsize=18)
axes[0,1].set_xlim([14, 50])
axes[0,1].set_ylim([0, .34])
axes[0,1].plot(df2.ageg, df2['Actual'],lw=1.25)
axes[0,1].plot(df2.ageg, df2['rural actual'],lw=1.25)
axes[0,1].plot(df2.ageg, df2['urban actual'],lw=1.25)

#Himachal pradesh

axes[0,2].set_xlim([14, 50])
axes[0,2].set_ylim([0, .34])
axes[0,2].plot(df3.ageg, df3['Actual'],lw=1.25)
axes[0,2].plot(df3.ageg, df3['rural actual'],lw=1.25)
axes[0,2].plot(df3.ageg, df3['urban actual'],lw=1.25)

#Gujarat

axes[2,2].set_xlim([14, 50])
axes[2,2].set_ylim([0, .34])
axes[2,2].plot(df4.ageg, df4['Actual'],lw=1.25)
axes[2,2].plot(df4.ageg, df4['rural actual'],lw=1.25)
axes[2,2].plot(df4.ageg, df4['urban actual'],lw=1.25)

#Kerala

axes[2,0].set_xlim([14, 50])
axes[2,0].set_ylim([0, .34])
axes[2,0].plot(df5.ageg, df5['Actual'],lw=1.25)
axes[2,0].plot(df5.ageg, df5['rural actual'],lw=1.25)
axes[2,0].plot(df5.ageg, df5['urban actual'],lw=1.25)

#Sikkim

axes[2,1].set_xlim([14, 50])
axes[2,1].set_ylim([0, .34])
axes[2,1].plot(df6.ageg, df6['Actual'],lw=1.25)
axes[2,1].plot(df6.ageg, df6['rural actual'],lw=1.25)
axes[2,1].plot(df6.ageg, df6['urban actual'],lw=1.25)

#Esitamted
#bihar

axes[1,0].set_xlim([14, 50])
axes[1,0].set_ylim([0, .34])
axes[1,0].plot(df1.ageg, df1['eatimated'],lw=1.25)
axes[1,0].plot(df1.ageg, df1['rural estimated'],lw=1.25)
axes[1,0].plot(df1.ageg, df1['urban estimated'],lw=1.25)

#punjab

axes[1,1].set_xlim([14, 50])
axes[1,1].set_ylim([0, .34])
axes[1,1].plot(df2.ageg, df2['eatimated'],lw=1.25)
axes[1,1].plot(df2.ageg, df2['rural estimated'],lw=1.25)
axes[1,1].plot(df2.ageg, df2['urban estimated'],lw=1.25)

#Himachal pradesh

axes[1,2].set_xlim([14, 50])
axes[1,2].set_ylim([0, .34])
axes[1,2].plot(df3.ageg, df3['eatimated'],lw=1.25)
axes[1,2].plot(df3.ageg, df3['rural estimated'],lw=1.25)
axes[1,2].plot(df3.ageg, df3['urban estimated'],lw=1.25)

#Gujarat

axes[3,2].set_xlim([14, 50])
axes[3,2].set_ylim([0, .34])
axes[3,2].plot(df4.ageg, df4['eatimated'],lw=1.25)
axes[3,2].plot(df4.ageg, df4['rural estimated'],lw=1.25)
axes[3,2].plot(df4.ageg, df4['urban estimated'],lw=1.25)

#Kerala

axes[3,0].set_xlim([14, 50])
axes[3,0].set_ylim([0, .34])
axes[3,0].plot(df5.ageg, df5['eatimated'],lw=1.25)
axes[3,0].plot(df5.ageg, df5['rural estimated'],lw=1.25)
axes[3,0].plot(df5.ageg, df5['urban estimated'],lw=1.25)


#Sikkim

axes[3,1].set_xlim([14, 50])
axes[3,1].set_ylim([0, .34])
axes[3,1].plot(df6.ageg, df6['eatimated'],lw=1.25)
axes[3,1].plot(df6.ageg, df6['rural estimated'],lw=1.25)
axes[3,1].plot(df6.ageg, df6['urban estimated'],lw=1.25)


# In[23]:


fig, axes = plt.subplots(nrows=4, ncols=3,figsize=(20,20))
fig = plt.figure()

axes[0,0].set_title('Bihar age specific fertility rates  ', fontsize=18)
axes[0,0].set_ylabel('Observed', fontsize=18)
axes[0,0].set_xlim([14, 50])
axes[0,0].set_ylim([0, .34])
axes[0,0].plot(df1.ageg, df1['Actual'],lw=1.25)
axes[0,0].plot(df1.ageg, df1['rural actual'],lw=1.25)
axes[0,0].plot(df1.ageg, df1['urban actual'],lw=1.25)



#punjab
axes[0,1].set_title('Punjab age specific fertility rates  ', fontsize=18)
axes[0,1].set_xlim([14, 50])
axes[0,1].set_ylim([0, .34])
axes[0,1].plot(df2.ageg, df2['Actual'],lw=1.25)
axes[0,1].plot(df2.ageg, df2['rural actual'],lw=1.25)
axes[0,1].plot(df2.ageg, df2['urban actual'],lw=1.25)

#Himachal pradesh

axes[0,2].set_xlim([14, 50])
axes[0,2].set_ylim([0, .34])
axes[0,2].plot(df3.ageg, df3['Actual'],lw=1.25)
axes[0,2].plot(df3.ageg, df3['rural actual'],lw=1.25)
axes[0,2].plot(df3.ageg, df3['urban actual'],lw=1.25)

#Gujarat

axes[2,2].set_xlim([14, 50])
axes[2,2].set_ylim([0, .34])
axes[2,2].plot(df4.ageg, df4['Actual'],lw=1.25)
axes[2,2].plot(df4.ageg, df4['rural actual'],lw=1.25)
axes[2,2].plot(df4.ageg, df4['urban actual'],lw=1.25)

#Kerala

axes[2,0].set_xlim([14, 50])
axes[2,0].set_ylim([0, .34])
axes[2,0].plot(df5.ageg, df5['Actual'],lw=1.25)
axes[2,0].plot(df5.ageg, df5['rural actual'],lw=1.25)
axes[2,0].plot(df5.ageg, df5['urban actual'],lw=1.25)

#Sikkim

axes[2,1].set_xlim([14, 50])
axes[2,1].set_ylim([0, .34])
axes[2,1].plot(df6.ageg, df6['Actual'],lw=1.25)
axes[2,1].plot(df6.ageg, df6['rural actual'],lw=1.25)
axes[2,1].plot(df6.ageg, df6['urban actual'],lw=1.25)

#Esitamted
#bihar

axes[1,0].set_xlim([14, 50])
axes[1,0].set_ylim([0, .34])
axes[1,0].plot(df1.ageg, df1['eatimated'],lw=1.25)
axes[1,0].plot(df1.ageg, df1['rural estimated'],lw=1.25)
axes[1,0].plot(df1.ageg, df1['urban estimated'],lw=1.25)

#punjab

axes[1,1].set_xlim([14, 50])
axes[1,1].set_ylim([0, .34])
axes[1,1].plot(df2.ageg, df2['eatimated'],lw=1.25)
axes[1,1].plot(df2.ageg, df2['rural estimated'],lw=1.25)
axes[1,1].plot(df2.ageg, df2['urban estimated'],lw=1.25)

#Himachal pradesh

axes[1,2].set_xlim([14, 50])
axes[1,2].set_ylim([0, .34])
axes[1,2].plot(df3.ageg, df3['eatimated'],lw=1.25)
axes[1,2].plot(df3.ageg, df3['rural estimated'],lw=1.25)
axes[1,2].plot(df3.ageg, df3['urban estimated'],lw=1.25)

#Gujarat

axes[3,2].set_xlim([14, 50])
axes[3,2].set_ylim([0, .34])
axes[3,2].plot(df4.ageg, df4['eatimated'],lw=1.25)
axes[3,2].plot(df4.ageg, df4['rural estimated'],lw=1.25)
axes[3,2].plot(df4.ageg, df4['urban estimated'],lw=1.25)

#Kerala

axes[3,0].set_xlim([14, 50])
axes[3,0].set_ylim([0, .34])
axes[3,0].plot(df5.ageg, df5['eatimated'],lw=1.25)
axes[3,0].plot(df5.ageg, df5['rural estimated'],lw=1.25)
axes[3,0].plot(df5.ageg, df5['urban estimated'],lw=1.25)


#Sikkim

axes[3,1].set_xlim([14, 50])
axes[3,1].set_ylim([0, .34])
axes[3,1].plot(df6.ageg, df6['eatimated'],lw=1.25)
axes[3,1].plot(df6.ageg, df6['rural estimated'],lw=1.25)
axes[3,1].plot(df6.ageg, df6['urban estimated'],lw=1.25)


# In[33]:


fig = plt.figure()
ax=fig.add_axes([0,0,1.8,1.8])


ax.plot(df1.Survey, df1['Married below 19'],lw=3)
ax.plot(df1.Survey, df1['Married above 19'],lw=3)
ax.set_title('True predction of higher education and age of first birth- recall values  ', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend(['first birth below 19 years', 'first birth above 19 years'],fontsize=18, loc=4)

ax2 = fig.add_axes([0.2,0.7,.9,.7])
ax2.plot(df1.Survey, df1['TFR'])


plt.yticks(fontsize=14)
plt.xticks(fontsize=12)

ax2.set_title('Total Fertility Rates', fontsize=18)


# In[34]:


fig = plt.figure()
ax=fig.add_axes([0,0,1,1.8])
ax.plot(df1.Survey, df1['Litrracy Rate'],lw=3)
x = list(df1['Survey'])
y = list(df1['Litrracy Rate'])
plt.scatter(x,y,marker="*",s=130,edgecolors="blue",c="blue")


# In[34]:


fig = plt.figure()
ax=fig.add_axes([0,0,1.8,1.8])


ax.plot(df1.Survey, df1['Married below 19'],lw=3)
ax.plot(df1.Survey, df1['Married above 19'],lw=3)
ax.set_title('True predction of higher education and age of first birth- recall values  ', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend(['first birth below 19 years', 'first birth above 19 years'],fontsize=18, loc=4)

ax2 = fig.add_axes([0.2,0.7,.9,.7])
ax2.plot(df1.Survey, df1['TFR'])
plt.scatter(x,y,marker="*",s=130,edgecolors="black",c="red")
x = list(df1['Survey'])
y = list(df1['TFR'])
lab_2 = list(df1['TFR'])
for i, label in enumerate(lab_2):
    plt.annotate(label, (x[i], y[i]+0.05), fontsize=11)
    


plt.yticks(fontsize=14)
plt.xticks(fontsize=12)

ax2.set_title('Total Fertility Rates', fontsize=18)


# In[37]:


xls = pd.ExcelFile('Plots_2.xlsx')
df1 = pd.read_excel(xls, 'Table_1')
df1 = df1.iloc[0: , :]
df1.head()


# In[38]:


xls = pd.ExcelFile('Plots_3.xlsx')
df1 = pd.read_excel(xls, 'Age_specific')
df1 = df1.iloc[0: , :]
df1.head()


# In[39]:


xls = pd.ExcelFile('Plots.xlsx')
df2 = pd.read_excel(xls, 'Table_1')
df2 = df2.iloc[0: , :]
df2.head()


# In[48]:


fig = plt.figure()
ax=fig.add_axes([0,0,1.8,1.8])


ax.plot(df1.ageg4, df1['ASFR NFHS 4'],lw=3)
x = list(df1['ageg4'])
y = list(df1['ASFR NFHS 4'])
plt.scatter(x,y,marker="*",s=130,edgecolors="blue",c="blue")
ax.plot(df1.ageg4, df1['ASFR NFHS 3'],lw=3)
x = list(df1['ageg4'])
y = list(df1['ASFR NFHS 3'])
plt.scatter(x,y,marker="v",s=130,edgecolors="orange",c="orange")
ax.plot(df1.ageg4, df1['ASFR NFHS 2'],lw=3)
x = list(df1['ageg4'])
y = list(df1['ASFR NFHS 2'])
plt.scatter(x,y,marker="P",s=130,edgecolors="green",c="green")
ax.plot(df1.ageg4, df1['ASFR NFHS 1'],lw=3)
x = list(df1['ageg4'])
y = list(df1['ASFR NFHS 1'])
plt.scatter(x,y,marker="H",s=130,edgecolors="red",c="red")

ax.set_title('True predction of higher education and age of first birth- recall values  ', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend(['NFHS 4', 'NFHS 3', 'NFHS 2', 'NFHS 1'],fontsize=14, loc=1)


# In[ ]:




