#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[2]:


fake_data = pd.read_csv('Fake.csv')
true_data = pd.read_csv('True.csv')



# In[3]:


fake_data.head()


# In[4]:


#Target feature
fake_data['class']=0
true_data['class']=1


# In[5]:


fake_data.shape,true_data.shape


# In[6]:


# Extracting manual testing data from the tail of the datasets
fake_data_manual_testing = fake_data.tail(10)
for i in range(23480, 23470, -1):  # Adjust range indices as per your dataset
    fake_data.drop([i], axis=0, inplace=True)

true_data_manual_testing = true_data.tail(10)
for i in range(21416, 21406, -1):  # Adjust range indices as per your dataset
    true_data.drop([i], axis=0, inplace=True)


# In[7]:


fake_data.shape


# In[8]:


fake_data_manual_testing['class']=0
true_data_manual_testing['class']=1


# In[9]:


data_merge=pd.concat([fake_data,true_data],axis=0)
data_merge.head(10)


# In[10]:


data_merge.columns


# In[11]:


data=data_merge.drop(['title','subject','date'],axis=1)


# In[12]:


data


# In[13]:


data.isnull().sum()


# In[14]:


#random suffling
data=data.sample(frac=1) #frac =1 means 100% suffling


# In[15]:


data


# In[16]:


data.reset_index(inplace=True)
data.drop(['index'],axis=1,inplace=True) #here index is column so axis =1 must


# In[17]:


#function to process the text
def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," " ,text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text
    
    


# In[18]:


data['text']=data['text'].apply(wordopt)


# In[19]:


x=data['text']
y=data['class']


# In[20]:


x
y


# In[21]:


y


# In[22]:


#train_test_splict
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer object
vectorization = TfidfVectorizer()

# Fit the vectorizer on the training data and transform it
xv_train = vectorization.fit_transform(x_train)

# Transform the test data based on the training data
xv_test = vectorization.transform(x_test)


# In[24]:


from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
LR = LogisticRegression()

# Fit the model with the training data and labels (x_train, y_train)
LR.fit(xv_train, y_train)


# In[25]:


pred_lr=LR.predict(xv_test)


# In[26]:


LR.score(xv_test,y_test)


# In[27]:


print(classification_report(y_test,pred_lr))


# In[28]:


#same task for gd and randomforest
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(xv_train, y_train)



# In[29]:


pred_dt=DT.predict(xv_test)


# In[30]:


print(classification_report(y_test,pred_dt))


# In[31]:


#for gd_classifier
from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)


# In[32]:


pred_rf=DT.predict(xv_test)


# In[33]:


RF.score(xv_test,y_test)


# In[34]:


print(classification_report(y_test,pred_rf))


# In[39]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    # Predictions
    pred_LR = LR.predict(new_xv_test)  # Logistic Regression
    pred_DT = DT.predict(new_xv_test)  # Decision Tree
    pred_RF = RF.predict(new_xv_test)  # Random Forest

    # Display results
    print("\n\nLR Prediction: {} \nDT Prediction: {} \nRF Prediction: {}".format(
        output_lable(pred_LR[0]),
        output_lable(pred_DT[0]),
        output_lable(pred_RF[0])
    ))

# Input and call the function
news = str(input("Enter the news text: "))
manual_testing(news)


# In[ ]:




