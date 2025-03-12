#!/usr/bin/env python
# coding: utf-8

# import libraries and dataset

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[3]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[13]:


#crea, orientte a figure with two subplots ,stacked vertically
fig,axes=plt.subplots(2,1,figsize=(8, 6),gridspec_kw={'height_ratios':[1, 3]})

#plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["daily"], ax=axes[0], color='skyblue', width=0.5, orient='h')
axes[0].set_xlabel("daily levels")

#plot the histogram with kde curve in the second (bottom) subplot
sns.histplot(data1["daily"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("daily Levels")
axes[1].set_ylabel("Frequency")

# adjest layout for better spacing
plt.tight_layout()

#show the plot
plt.show


# In[15]:


plt.scatter(data1["sunday"], data1["daily"])


# In[16]:


# import numpy as np
# x = np.arrange(10)
# plt.plot(2 + 3 *X)
# plt.show()


# In[19]:


#coefficients
model.params


# In[ ]:




