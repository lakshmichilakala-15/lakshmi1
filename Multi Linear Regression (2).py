#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# Assumptions in multilinear Regression
# 1.Linearity:The relationship between the predictors(x) and the response(Y) is linear
# 2.Independence:Observations are independent pf each other
# 3.Homoscedasity:The reseduals (Y - Y_hat) exhibit constant variance at all levels of the predictor
# 4.Normal Distribiution of Error: The residuals of the model are normally distributed.
# 5.No Multicollinearity: The independent variable should not be too highly correlated with each other.
# **Vilotions of these assumption may lead to inefficiency in the regression parameters and unreliable prediction**

# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()

- MPG : Milege of the (Mile per Gallon) (This is Y-column to be predicted)
- HP : Horse Power of the car(X1 column)
- VOL : Volume of the car (size) (X2 column)
- SP : Top speed of the car (Miles per Hour)(X3 column)
- WT : Weight of the car (Pounds) (X4 Column)
    
# # EDA

# In[4]:


cars.info()


# In[5]:


cars.isna().sum()


# **observatons about info(), missing values**
#     -there are no missing values
#     -there are 81 observations 
#     -the data types of the columns are also relevent and valid
# 

# In[6]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[7]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[8]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[9]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[10]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# **OBSERVATIONS**
# >THERE ARE SOME EXTREME VALUES OBSERVED IN TOWARDS RIGHT TAIL OF HP AND SP DISTRIBUTIONS
# >IN VOL AND WT A FEW OUTLIERS OBSERVED 
# >THE EXTREME VALUES OF CARS DATA MAY HAVE COME FROM THE SPECIALLY DESIGNED NATURE OF  CARS
# >AS THIS MULTI-DIMENSIONAL DATRA, THE OUTLIERS WITH RESPECT TO SPATIAL DIMENSIONS MAY HAVE TO BE CONSIDERED
# 

# In[11]:


cars[cars.duplicated()]


# In[12]:


cars.corr()


# HP has positive corelation with sp
# vol and wt has positive corelation
# -Between x and y all the x variable are showing moderate to high correlation strengths,highest being between HP and MPG
# -Therfore this data set qualifies for building a multiple linear regression model to predict MPG
# -among xx colums(x1,x2,x3 and x4),some very high correlation strength observed between SP vs HP,VOL vsWT
# -The high correlation among x columns is not desirable as it might lead to multicollinearity problem
# 

# In[15]:


model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[17]:


model1.summary()


# ŷ= 30.677 +
# + 0.4006 (WT)+
# -0.3361 (VOL)+
# + 0.3956(SP)+
# -0.2054(HP)

# **OBSERVATION  FROM MODEL SUMMARY**
# + THE R-squared and adjusted R-squared values are good and about 75% of variability in Y is explained by X columns
# + The probability value with respect to F-statistic is close to zero, indicating that all or someof X columns are significant
# + the P-value for VOL and WT are higher than 5% indicating issue among themselves,Which need to be further explored

# **PERFORMANCE METRICS FOR MODEL1**

# In[18]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[19]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[ ]:




