#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import numpy Library
import numpy as np


# In[2]:


# dir(np)


# In[4]:


# create 1D numpy array
x = np.array([45,67,57,60])
print(x)
print(type(x))
print(x.dtype)


# In[5]:


# verify the data type in raray
x = np.array([45,67,57,9.8])
print(x)
print(type(x))
print(x.dtype)


# In[7]:


# verify the data type
x = np.array(["A",67,57,9.8])
print(x)
print(type(x))
print(x.dtype)


# In[9]:


# create an 2D array
a2 = np.array([[20,40],[30,60]])
print(a2)
print(type(a2))
print(a2.shape)


# In[10]:


# Reshaping an array using reshape()
a = np.array([10,20,30,40])
b = a.reshape(2,2)
print(b)
print(b.shape)


# In[12]:


# Create an array with arange()
c = np.arange(3,10)
print(c)
type(c)


# In[14]:


# Use of around()
d = np.array([1.3467, 3.10987,4.91236])
print(d)
np.around


# In[15]:


a1 = np.array([[3,4,5,8],[7,2,8,np.NaN]])
print(a1)
a1.dtype


# In[16]:


a4 = np.array([[3,4,5],[7,2,8],[9,1,6],[10,9,18]])
print(a4)


# In[19]:


a3 = np.array([[3,4,5],[7,2,8],[9,1,6]])
print(a3)


# In[18]:


# print the index position of max element
print(np.argmax(a3,axis=1))
print(np.argmax(a3,axis=0))


# In[21]:


# print the max value element
print(np.amax(a3, axis =1))
print(np.amax(a3, axis =0))


# In[ ]:




