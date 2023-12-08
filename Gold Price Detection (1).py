#!/usr/bin/env python
# coding: utf-8

# In[55]:


#Importing libraries

import pandas as pd
import numpy as n
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[ ]:





# In[82]:


data=pd.read_csv('Gold Price Predictor Data.csv')


# In[57]:


data.head()
data.tail()


# In[58]:


data.shape


# In[59]:


data.info()


# In[60]:


data.isnull().sum()


# In[61]:


data.describe()


# In[62]:


correlation=data.corr()


# In[63]:


sns.heatmap(correlation,cbar=True,square=True,fmt='.if',annot=True,annot_kws={'size:8'})


# In[ ]:


sns.scatterplot(x='GDX_Close',y='Close',data=data)


# In[64]:


print(correlation)


# In[65]:


sns.displot(data['GDX_Close'])


# In[66]:


X=data.drop(['Date','GDX_Close'],axis=1)
Y=data['GDX_Close']


# In[67]:


print(X)
print(Y)


# In[68]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


# # Model Training:Random Forest Regressor

# In[69]:


regressor=RandomForestRegressor(n_estimators=100)


# In[70]:


regressor.fit(X_train,Y_train)


# In[71]:


test_data_prediction=regressor.predict(X_test)


# In[72]:


print(test_data_prediction)


# # For comparing model:R squared error
# 

# In[73]:


error_score=metrics.r2_score(Y_test,test_data_prediction)


# In[74]:


print(error_score)


# # Comparing Actual and Predicted Values

# In[75]:


Y_test=list(Y_test)


# In[80]:


plt.plot(Y_test,color='Red',label='Actual Values')
plt.plot(test_data_prediction,color='green',label='Predicted Values')
plt.title('Actual Vs Predicted Price')
plt.xlabel('No of Values')
plt.ylabel('Gold Price')
plt.legend()
plt.show()


# In[ ]:




