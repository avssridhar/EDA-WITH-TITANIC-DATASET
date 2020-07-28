#!/usr/bin/env python
# coding: utf-8

# # EXLPLORATORY DATA SET OF TITANIC DATA SET
# 

# In[2]:


#more than 60npercent of time goes into EDA
#GO WITH LIFECYCLE OF DATA SCYCLE 
#gO AND HANDLE MISSING DATA
#seaborn,numpy,pandas,matplotlab libraries are important


# In[3]:


#we will use titanic dataset
#we try to predict classification of surivival and deceased
#we implement logistic regression


# In[4]:


#importlibrries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


#data
data=pd.read_csv('C:/Users/sridhar/Downloads/11657_16098_bundle_archive/train.csv')


# In[10]:


data.head()


# In[12]:


#missingddata
data.isnull()
#if it is true that value is null


# In[15]:


#this will become difficult to see what are actuallty null
#and we cannot see all
#hence we should visualize 
sns.heatmap(data.isnull(),yticklabels=False,cmap='viridis')
#all null value are shown in yellow color


# In[19]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=data)


# In[21]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=data,palette='RdBu_r')


# In[22]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=data,palette='rainbow')


# In[24]:


sns.distplot(data['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[25]:


data['Age'].hist(bins=30,color='darkred',alpha=0.3)


# In[27]:


sns.countplot(x='SibSp',data=data)


# In[28]:


data['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[29]:


#till now we have seen what is in the dataset and what are relations 
#now lets clean the data


# In[30]:


#datacleaning
#removing null value
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=data,palette='winter')


# In[31]:


#we cans ee if passengers in the higher classes tend to be o;der 
#this makes sense 
#we will use these average age values to impute based on Pclass for Age


# In[46]:


def impute_age(cols):
    Age=cols[0]
    Pclasss=cols[1]
    if pd. null(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
        
    else:
            return Age 
        


# In[47]:


#NOE APPLY THE FUNCTION
data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1)


# In[48]:


#ACTUALLY THE ABOVE FUNCTION REPLACES ALL THE NULL VALUES IN 
#AGE AND Pclass column


# In[49]:


#when you type the following heatmap to check the nullvalues you wont find any null values 
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[50]:


####actualt there should not be any yellowsince the fucntion hasnt run
#you got the wrong output


# In[51]:


#drop the cabin column and the row that is embarked that is NaN


# In[52]:


data.drop('Cabin',axis=1,inplace=True)


# In[53]:


data.head()


# In[54]:


data.dropna(inplace=True)


# In[55]:


data.head()


# In[56]:


#converting categorical features 
#we convert categorical variables to dummy variables using pandas
#otherwise our ml model wont be able to directly take in those features as inputs 


# In[57]:


data.info()


# In[60]:


pd.get_dummies(data['Embarked'],drop_first=True).head()


# In[63]:


#CONVERTING CATEGORICAL VARIBLES 
#DROPPING THECATEGORICLA VARIABLES 
sex=pd.get_dummies(data['Sex'],drop_first=True)
embark=pd.get_dummies(data['Embarked'],drop_first=True)


# In[65]:


data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[66]:


data.head()


# In[67]:


#asyou can notice all categorical columns are removed |


# In[68]:


data=pd.concat([data,sex,embark],axis=1)


# In[69]:


data.head()


# In[70]:


#building a logistic regression model 


# In[72]:


#train test split
data.drop('Survived',axis=1).head()


# In[73]:


data['Survived'].head()


# In[74]:


from sklearn.model_selection import train_test_split


# In[77]:


X_train,X_test,y_train,y_test=train_test_split(data.drop('Survived',axis=1),
                                               data['Survived'],test_size=0.30,
                                               random_state=101)


# In[78]:


#training and predicting 


# In[80]:


from sklearn.linear_model import LogisticRegression


# In[81]:


logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)


# In[82]:


predictions=logmodel.predict(X_test)


# In[84]:


from sklearn.metrics import confusion_matrix


# In[86]:


accuracy=confusion_matrix(y_test,predictions)


# In[87]:


accuracy


# In[88]:


from sklearn.metrics import accuracy_score


# In[90]:


accuracy=accuracy_score(y_test,predictions)


# In[91]:


accuracy


# In[92]:


predictions


# In[ ]:




