#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[5]:


df = pd.read_excel("C:/Users/Sid/Desktop/OPER 655/Code/Preprocessing/Cleaned data w pos.xlsx")
df


# In[6]:


titles = df['Title']
titles.to_csv('Titles.csv', index = False)


# In[9]:


from sklearn.preprocessing import LabelEncoder

categoricals = ["Season posted", "Time of Day", "Category"]
label_encoders = {}  # To store the encoder for each column

for col in categoricals:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])
    # Store the encoder for later inspection
    label_encoders[col] = label_encoder

# To see the transformation for a specific column:
for col, encoder in label_encoders.items():
    print(f"Mapping for column '{col}':")
    for class_, label in zip(encoder.classes_, range(len(encoder.classes_))):
        print(f"{class_} -> {label}")
    print()


# # Word2Vec

# In[6]:


import gensim.downloader as api

# Load pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")  # 300-dimensional embeddings

# Function to get embeddings for each word in a title
def get_word_embeddings(title, model):
    words = title.split()  # Split the title into words
    word_embeddings = []
    for word in words:
        try:
            embedding = model[word]  # Get the word vector from the model
            word_embeddings.append(embedding)
        except KeyError:
            # If the word isn't in the model's vocabulary, append a zero vector
            word_embeddings.append(np.zeros(300))  # Adjust 300 to match the model's vector size
    return np.array(word_embeddings)

# Apply the function to the 'Title' column in your DataFrame
df_w2v = df.copy()
df_w2v['Title_embeddings'] = df_w2v['Title'].apply(lambda x: get_word_embeddings(x, model))
df_w2v = df_w2v.drop(columns = ['Title'])
df_w2v.head()
df_w2v.to_csv("W2V.csv", index=False)


# # BERT

# In[8]:


import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(title, tokenizer, model):
    # Tokenize and encode the title text
    inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True, max_length=50)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the embeddings for the [CLS] token (or average the entire sequence)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Average over the sequence
    return embeddings

# Apply BERT embeddings to each title
df_bert = df.copy()
df_bert['Title_embeddings'] = df_bert['Title'].apply(lambda x: get_bert_embeddings(x, tokenizer, model))
df_bert = df_bert.drop(columns = ['Title'])
df_bert.head()

df_bert.to_csv("Bert.csv", index=False)


# # Glove

# In[10]:


model = api.load("glove-wiki-gigaword-100")

def get_glove_embeddings(title, model):
    words = title.split()
    word_embeddings = []
    for word in words:
        try:
            embedding = model[word]  # Get the word vector from the GloVe model
            word_embeddings.append(embedding)
        except KeyError:
            # If the word isn't in the model's vocabulary, append a zero vector
            word_embeddings.append(np.zeros(100))  # Adjust the size if needed
    return np.array(word_embeddings)

df_glove = df.copy()
df_glove['Title_embeddings'] = df_glove['Title'].apply(lambda x: get_glove_embeddings(x, model))
df_glove = df_glove.drop(columns = ['Title'])
df_glove.head()

df_glove.to_csv("Glove.csv", index=False)


# # TF-IDF

# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=50)  # Set the max number of features to use
X_tfidf = tfidf_vectorizer.fit_transform(df['Title'])
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df_tf = pd.concat([df, tfidf_df], axis=1)
df_tf = df_tf.drop(columns = ['Title'])
df_tf.head()

df_tf.to_csv("TF-IDF.csv", index=False)

