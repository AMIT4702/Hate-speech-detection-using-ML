#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[2]:


import nltk
import re
from nltk.corpus import stopwords
stopword=set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")


# In[3]:


data=pd.read_csv("C:/Users/Gaur_Vishal/Downloads/twitter_data.csv")
print(data.head())


# In[4]:


data["labels"]=data["class"].map({0:"Hate Speech", 1:"Offensive Speech", 2:"No Hate and Offensive Speech"})


# In[5]:


data=data[["tweet","labels"]]


# In[6]:


data.head()


# In[7]:


import re


# In[8]:


def clean (text):
     text = str(text).lower()
     text = re.sub('[.?]', '', text) 
     text = re.sub('https?://\S+|www.\S+', '', text)
     text = re.sub('<.?>+', '', text)
     text = re.sub(r'[^\w\s]','',text)
     text = re.sub('\n', '', text)
     text = re.sub('\w\d\w', '', text)
     text = [word for word in text.split(' ') if word not in stopword]
     text=" ".join(text)
     text = [stemmer. stem(word) for word in text. split(' ')]
     text=" ".join(text)
     return text
data["tweet"] = data["tweet"].apply(clean)


# In[9]:


x=np.array(data["tweet"])
y=np.array(data["labels"])


# In[10]:


cv=CountVectorizer()


# In[11]:


X = cv.fit_transform(x)


# In[12]:



X_train,X_text,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[13]:



model= DecisionTreeClassifier()


# In[14]:


model.fit(X_train,y_train)


# In[15]:


y_pred=model.predict(X_text)


# In[16]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='weighted')

recall = recall_score(y_test, y_pred, average='weighted')

f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[21]:


user = str(input("enter the tweet : "))
i= user
i = cv.transform([i]).toarray()
print(model.predict((i)))


# In[ ]:





# In[ ]:





# In[ ]:




