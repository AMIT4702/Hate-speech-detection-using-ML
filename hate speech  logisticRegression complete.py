#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay


# In[8]:


tweet_df = pd.read_csv('C:/Users/Gaur_Vishal/Downloads/twitter_data.csv')
tweet_df.head()


# In[9]:


tweet_df.info()


# In[1]:



'''text =' '.join([word for word in tweet_df['tweet']])

plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in non hate tweets', fontsize = 19)
plt.show()

#print(text)'''


# In[11]:


print(tweet_df['tweet'].iloc[0],"\n")
print(tweet_df['tweet'].iloc[1],"\n")
print(tweet_df['tweet'].iloc[2],"\n")
print(tweet_df['tweet'].iloc[3],"\n")
print(tweet_df['tweet'].iloc[4],"\n")


# In[12]:


#creating a function to process the data
'''def data_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet, flags = re.MULTILINE)
    tweet = re.sub(r'\@w+|\#','', tweet)
    tweet = re.sub(r'[^\w\s]','',tweet)
    tweet = re.sub(r'ð','',tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_tweets)'''
def data_processing(tweet):
    if isinstance(tweet, list):
        tweet = ' '.join(tweet)  # Assuming the list contains strings
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@w+|\#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'ð', '', tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_tweets)


# In[13]:


#tweet_df.tweet = tweet_df['tweet'].apply(data_processing)
tweet_df['tweet'] = tweet_df['tweet'].apply(data_processing)


# In[14]:


tweet_df = tweet_df.drop_duplicates('tweet')


# In[15]:


vect = TfidfVectorizer(ngram_range=(1,2)).fit(tweet_df['tweet'])


#lemmatizer = WordNetLemmatizer()
#def lemmatizing(data):
 #   tweet = [lemmarizer.lemmatize(word) for word in data]
  #  return data
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatizing(data):
    tweet = [lemmatizer.lemmatize(word) for word in data]
    return tweet


# In[16]:


#tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: lemmatizing(x))
tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: lemmatizing(x))


# In[17]:


print(tweet_df['tweet'].iloc[0],"\n")
print(tweet_df['tweet'].iloc[1],"\n")
print(tweet_df['tweet'].iloc[2],"\n")
print(tweet_df['tweet'].iloc[3],"\n")
print(tweet_df['tweet'].iloc[4],"\n")


# In[18]:


tweet_df.info()


# In[19]:


tweet_df['hate_speech'].value_counts()


# In[20]:


fig = plt.figure(figsize=(5,5))
sns.countplot(x='hate_speech',data = tweet_df)


# In[21]:


'''import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 7))
colors = ("red", "gold")
wp = {'linewidth': 2, 'edgecolor': "black"}
tags = tweet_df['hate_speech'].value_counts()
explode = (0.1, 0)  # Adjust the explode parameter here, it must have the same length as the number of unique values
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, startangle=90,
           wedgeprops=wp, explode=explode,hate_speech = ' ')
plt.title('Distribution of sentiments')
plt.show()'''


# In[34]:


text = ' '.join([word for word in non_hate_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in non hate tweets', fontsize = 19)
plt.show()


# In[ ]:


non_hate_tweets = tweet_df[tweet_df.hate_speech==0]
non_hate_tweets.head()


# In[ ]:





# In[ ]:


neg_tweets = tweet_df[tweet_df.hate_speech == 1]
neg_tweets.head()



# In[ ]:


pip install --upgrade Pillow


# In[ ]:


feature_names = vect.get_feature_names()
print("number of features: {}\n".format(len(feature_names)))
print("first 20 features: \n {}".format(feature_names[:20]))


# In[23]:


#X = tweet_df['tweet']
#Y = tweet_df['hate_speech']
#X = vect.transform(X)
X = [' '.join(tweet) for tweet in tweet_df['tweet']]
Y = tweet_df['hate_speech']
X = vect.transform(X)


# In[24]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


# In[25]:


print("size of x_train:",(x_train.shape))
print("size of y_train:",(y_train.shape))
print("size of x_test:",(x_train.shape))
print("size of y_test:",(y_train.shape))


# In[26]:


#logreg = LogisticRegression()


# In[27]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_predict = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_predict, y_test)
print("Test accuarcy: {:.2f}%".format(logreg_acc*100))


# In[28]:


print(confusion_matrix(y_test, logreg_predict))
print("\n")
print(classification_report(y_test, logreg_predict))


# In[32]:


style.use('classic')
cm = confusion_matrix(y_test, logreg_predict, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot()


# In[29]:


from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[30]:


param_grid = {'C':[100, 10, 1.0, 0.1, 0.01], 'solver' :['newton-cg', 'lbfgs','liblinear']}
grid = GridSearchCV(LogisticRegression(), param_grid, cv = 5)
grid.fit(x_train, y_train)
print("Best Cross validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)


# In[31]:


y_pred = grid.predict(x_test)


# In[32]:


logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


# In[33]:


print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))


# In[ ]:




