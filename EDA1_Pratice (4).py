#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


# Drop duplicated rows
data.drop_duplicates(keep='first', inplace = True)
data


# In[4]:


# Change column names (Rename the columns)
data.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data


# In[5]:


# Display data1 missinmg values count in each column using isnull().sum()
data.isnull().sum()


# In[6]:


# visualize data1 missing values using heat map

cols = data.columns
colors = ['black', 'yellow']
sns.heatmap(data[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[7]:


# Find the mean and median values of each numeric
#Imputation of missing value with median
median_ozone = data["Ozone"].median()
mean_ozone = data["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[8]:


# Replace the Ozone missing values with median value
data['Ozone'] = data['Ozone'].fillna(median_ozone)
data.isnull().sum()


# In[9]:


# Replace the Ozone missing values with median value
data['Ozone'] = data['Ozone'].fillna(mean_ozone)
data.isnull().sum()


# In[10]:


median_solar = data["Solar"].median()
mean_solar = data["Solar"].mean()
print("Median of Solar: ", median_ozone)
print("Mean of Solar: ", mean_ozone)


# In[11]:


data['Solar'] = data['Solar'].fillna(median_ozone)
data.isnull().sum()


# In[12]:


print(data["Weather"].value_counts())
mode_weather = data["Weather"].mode()[0]
print(mode_weather)


# In[13]:


data["Weather"] = data["Weather"].fillna(mode_weather)
data.isnull().sum()


# In[14]:


print(data["Month"].value_counts())
mode_month = data["Month"].mode()[0]
print(mode_month)


# In[15]:


data["Month"] = data["Month"].fillna(mode_month)
data.isnull().sum()


# # Detection of outliers in the columns

# Method1:Using histograms and box plots

# In[16]:


#Create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

#plot the boxplot in the first (top) subplot
sns.boxplot(data=data["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone levels")

#plot the histogram with KDE curve in the second (bottom) subplot
sns.histplot(data=data["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone levels")
axes[1].set_xlabel("Frequency")

#Adjust layout for better spacing
plt.tight_layout()

#show the plot
plt.show()



# # Observations

# THe ozone column has extreme values beyond 81 as seen from box plot
# The same is confirmed from the below right-slewed histrogram

# In[17]:


plt.figure(figsize=(6,2))
plt.boxplot(data["Ozone"], vert=False)


# In[18]:


# Extract outliers from boxplot for Ozone column
plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']] 


# In[19]:


data["Ozone"].describe()


# In[20]:


mu = data["Ozone"].describe()[1]
sigma = data["Ozone"].describe()[2]

for x in data["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# Quantitle-Quantile plot for detection of outlier

# In[21]:


import scipy.stats as stats

# Create Q-Q plot
plt.figure(figsize=(8,6))
stats.probplot(data["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# In[22]:


# Create a figure for violin plot

sns.violinplot(data=data["Ozone"], color='lightgreen')
plt.title("Violin Plot")


# # Other visualisations that could help understand the data

# In[23]:


# Create a figure for violin plot

sns.violinplot(data=data["Ozone"], color='lightgreen')
plt.title("Violin Plot")

#show the plot
plt.show()


# In[24]:


sns.swarmplot(data=data, x = "Weather", y = "Ozone",color="orange",palette="Set2", size=6)


# In[25]:


sns.stripplot(data=data, x = "Weather", y = "Ozone",color="orange",palette="Set2", size=6, jitter = True)


# In[26]:


sns.kdeplot(data=data["Ozone"], fill=True, color="blue")
sns.rugplot(data=data["Ozone"], color="black")


# In[27]:


# Category wise boxplot for ozone
sns.boxplot(data = data, x = "Weather", y="Ozone")


# Correlation coefficient and pair plots

# In[28]:


plt.scatter(data["Wind"], data["Temp"])


# In[29]:


# Compute peearson correlation coefficient
#between wind speed and Temperature
data["Wind"].corr(data["Temp"])


# In[30]:


# Read all numeric columns into a new table
data_numeric = data.iloc[:,[0,1,2,6]]
data_numeric


# In[31]:


# print correlation coefficients for all the above columns
data_numeric.corr()


# In[34]:


### ObservationsT
- The highest correlation strength between Ozone and Temperature(0.59708)
- The next  highest correlation strength is observed between Ozone and wind(-0.523708)
- The next  highest correlation strength is observed between Ozone and Temp(-0.442228)


# In[35]:


sns.pairplot(data_numeric)


# Transformation

# In[39]:


import pandas as pd

data2 = pd.get_dummies(data, columns=['Weather'])
data2


# In[40]:


import pandas as pd

data2 = pd.get_dummies(data, columns=['Month','Weather'])
data2


# In[41]:


data_numeric.values


# In[45]:


#Normalization of the data
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

array = data_numeric.values

scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(array)

#tranformed data
set_printoptions(precision=2)
print(rescaledX[0:10,:])


# In[ ]:




