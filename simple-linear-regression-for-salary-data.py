#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all the lib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


# read the dataset using pandas
data = pd.read_csv('Salary_Data.csv')


# In[3]:


# This displays the top 5 rows of the data
data.head()


# In[4]:


# Provides some information regarding the columns in the data
data.info()


# In[5]:


# this describes the basic stat behind the dataset used 
data.describe()


# In[7]:


# These Plots help to explain the values and how they are scattered

plt.figure(figsize=(12,6))
sns.pairplot(data,x_vars=['YearsExperience'],y_vars=['Salary'],size=7,kind='scatter')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.show()


# In[8]:


# Cooking the data
X = data['YearsExperience']
X.head()


# In[9]:


# Cooking the data
y = data['Salary']
y.head()


# In[10]:


# Import Segregating data from scikit learn
from sklearn.model_selection import train_test_split


# In[11]:


# Split the data for train and test 
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100)


# In[12]:


# Create new axis for x column
X_train = X_train[:,np.newaxis]
X_test = X_test[:,np.newaxis]


# In[13]:


# Importing Linear Regression model from scikit learn
from sklearn.linear_model import LinearRegression


# In[14]:


# Fitting the model
lr = LinearRegression()
lr.fit(X_train,y_train)


# In[15]:


# Predicting the Salary for the Test values
y_pred = lr.predict(X_test)


# In[16]:


# Plotting the actual and predicted values

c = [i for i in range (1,len(y_test)+1,1)]
plt.plot(c,y_test,color='r',linestyle='-')
plt.plot(c,y_pred,color='b',linestyle='-')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Prediction')
plt.show()


# In[17]:


# plotting the error
c = [i for i in range(1,len(y_test)+1,1)]
plt.plot(c,y_test-y_pred,color='green',linestyle='-')
plt.xlabel('index')
plt.ylabel('Error')
plt.title('Error Value')
plt.show()


# In[18]:


# Importing metrics for the evaluation of the model
from sklearn.metrics import r2_score,mean_squared_error


# In[19]:


# calculate Mean square error
mse = mean_squared_error(y_test,y_pred)


# In[20]:


# Calculate R square vale
rsq = r2_score(y_test,y_pred)


# In[21]:


print('mean squared error :',mse)
print('r square :',rsq)


# In[22]:


# Just plot actual and predicted values for more insights
plt.figure(figsize=(12,6))
plt.scatter(y_test,y_pred,color='r',linestyle='-')
plt.show()


# In[23]:


# Intecept and coeff of the line
print('Intercept of the model:',lr.intercept_)
print('Coefficient of the line:',lr.coef_)


# ![](http://)Then it is said to form a line with
# # y = 25202.8 + 9731.2x

# In[24]:


import pickle
pickle.dump(lr, open('app.pkl', 'wb'))


# In[ ]:




