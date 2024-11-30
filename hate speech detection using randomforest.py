#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[3]:


data = pd.read_csv(r"D:\OneDrive\Desktop\Annu\train.csv")
data.head()


# In[4]:


X = data["tweet"]
y = data["label"]


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[6]:


vectorizer = TfidfVectorizer(max_features=5000)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)


# In[7]:


rf_model = RandomForestClassifier(n_estimators=100)  
rf_model.fit(X_train_features, y_train)


# In[8]:


y_pred = rf_model.predict(X_test_features)


# In[9]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[10]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[11]:


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[ ]:




