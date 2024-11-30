#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# In[8]:


data = pd.read_csv("D:/OneDrive/Desktop/Annu/train.csv")
data.head()
data


# In[9]:


X = data['tweet']
y = data['label']
X,y


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# In[10]:


nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)


# In[12]:


y_pred = nb_classifier.predict(X_test_tfidf)
y_pred


# In[13]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:




