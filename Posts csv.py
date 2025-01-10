#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[12]:


# Load Excel
df = pd.read_excel("sid_reddit_database_with_features_and_content.xlsx", sheet_name="Sheet1")

# Filter posts
df_posts = df[df["Post/Reply"] == "Post"]

df_posts.head()


# In[15]:


df_posts.tail()


# In[14]:


df_posts.shape


# In[17]:


# Drop irrelevant features
df1 = df_posts.drop(columns = ["Post/Reply","Post ID", "Parent ID", "Comment ID", "Body HTML", "Body", "Body Length", "Body Word Count", "Reply Length", "Reply Word Count", "Username","Content"])
df1.head()


# In[19]:


# Define a function to assign seasons based on the month
def season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Convert your date column to datetime
df1['Created'] = pd.to_datetime(df1['Created'])

df1['Season posted'] = df1['Created'].apply(season)

df1.tail()


# In[22]:


# Repeat for time of day
def time_of_day(time):
    if 6 <= time < 12:
        return "Morning"
    elif 12 <= time < 17:
        return "Afternoon"
    elif 17 <= time < 22:
        return "Evening"
    else:
        return "Night"

df1['Time of Day'] = df1['Created'].dt.hour.apply(time_of_day)

df1.tail()


# In[24]:


df1 = df1.drop(columns = ["Created"])
df1.tail()               


# In[26]:


df1['Question'] = df1['Title'].apply(lambda x: 1 if '?' in str(x) else 0)
df1.head()


# In[2]:


df1['Quote'] = df1['title'].apply(lambda x: 1 if "'" in x or '"' in x else 0)
df1.head()


# In[32]:


df1 = df1.drop(columns = ["Quote"])
df1.head()


# In[43]:


# Write filtered df to a csv
df1.to_excel("Posts.xlsx", index=False)


# In[ ]:




