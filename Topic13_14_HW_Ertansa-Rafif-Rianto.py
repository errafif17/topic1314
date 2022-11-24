#!/usr/bin/env python
# coding: utf-8

# ## Multivariate Linear Regression

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics


# In[2]:


#Check the dataframe
df = pd.read_csv('train.csv')
df


# In[3]:


#Check the dataframe columns
df.columns


# In[4]:


#Use one-hot encoding to change categorical data to numerical data
df1 = pd.get_dummies(df)
df1


# In[5]:


#Fill NaN with number 0
df2 = df1.fillna(0)
#Check for NaN
obj = df2.isnull().sum()
for key,value in obj.iteritems():
    print(key,",",value)


# In[6]:


#check on dataframe info
#we see that there is no 'object' anymore on dataframe, only numerical data such as 'float64', 'int64', 'uint8'
df2.info()


# In[7]:


X=df2['OverallQual']
y=df2['SalePrice']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=1)


# In[8]:


X_train.shape


# In[9]:


y_train.shape


# In[13]:


# create linear regression object
reg = linear_model.LinearRegression()

#Fix shape of X_train and y_train
X_train = X_train.values.reshape(876, 1)
y_train = y_train.values.reshape(876, 1)

# train the model using the training sets
reg.fit(X_train, y_train)


# In[14]:


# regression coefficients
print('Coefficients: ', reg.coef_)

# regression intercept
print('Intercept: ', reg.intercept_)


# In[15]:


X_test.shape


# In[16]:


y_test.shape


# In[18]:


#Fix shape of the X_test and y_test
X_test = X_test.values.reshape(584, 1)
y_test = y_test.values.reshape(584, 1)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))
 
# plot for residual error
 
## setting plot style
plt.style.use('fivethirtyeight')
 
## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
 
## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
 
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
 
## plotting legend
plt.legend(loc = 'upper right')
 
## plot title
plt.title("Residual errors")
 
## method call for showing the plot
plt.show()


# In[19]:


y_pred = reg.predict(X_test)


# In[20]:


# MAE
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)


# In[21]:


# MSE
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[22]:


# RMSE
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred, squared=False)


# In[23]:


#Corelation between 3 features and 'SalePrice' as target
df2[['OverallQual', 'GrLivArea', 'GarageArea', 'SalePrice']].corr()['SalePrice'][:]


# ## Classification Models

# In[24]:


df4 = pd.read_csv('heart.csv')
df4


# In[25]:


#Use one-hot encoding to change categorical data to numerical data
df5 = pd.get_dummies(df4)
df5


# In[26]:


#Check for NaN
obj = df5.isnull().sum()
for key,value in obj.iteritems():
    print(key,",",value)


# In[27]:


X1=df5['Age']
y1=df5['HeartDisease']
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.4,
                                                    random_state=1)


# In[28]:


X_train1.shape


# In[29]:


y_train1.shape


# In[31]:


#Fix shape of X_train
X_train1 = X_train1.values.reshape(550, 1)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(X_train1, y_train1)


# In[32]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)

classifier_rf.fit(X_train1, y_train1)


# In[33]:


#Hyperparameter tuning
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}


# In[34]:


from sklearn.model_selection import GridSearchCV

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")

grid_search.fit(X_train1, y_train1)


# In[35]:


grid_search.best_score_


# In[36]:


rf_best = grid_search.best_estimator_
rf_best.fit(X_train1, y_train1)


# In[37]:


X_test1.shape


# In[46]:


#Fix shape of X_test
X_test1 = X_test1.reshape(368,1)
#Evaluate the result with confusion matrix, classification report, and AUC
y_lr = lr.predict(X_test1)
y_rf = rf_best.predict(X_test1)


# In[47]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test1, y_lr))
print(confusion_matrix(y_test1, y_rf))


# In[48]:


#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test1, y_lr))
print(classification_report(y_test1, y_rf))


# In[49]:


#AUC
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test1, y_lr, pos_label=1) # pos_label: positive label
print(auc(fpr, tpr))
fpr, tpr, thresholds = roc_curve(y_test1, y_rf, pos_label=1) # pos_label: positive label
print(auc(fpr, tpr))


# In[50]:


#Whick model is better?
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test1, y_lr))
print(accuracy_score(y_test1, y_rf))

from sklearn.metrics import precision_score
print(precision_score(y_test1, y_lr, average='macro'))
print(precision_score(y_test1, y_rf, average='macro'))

from sklearn.metrics import recall_score
print(recall_score(y_test1, y_lr, average='macro'))
print(recall_score(y_test1, y_rf, average='macro'))


# In[ ]:


#From data above Random Forest has the greatest amout of precision, accuracy, and true positive rate (recall)

