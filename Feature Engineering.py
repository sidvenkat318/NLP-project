#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import spacy
from collections import Counter
import re


# In[36]:


get_ipython().system('python -m spacy download en_core_web_sm')
# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# In[37]:


df = pd.read_excel("Posts.xlsx")
df = df.iloc[:,:-2]
df.head()


# In[38]:


# Function to calculate keyword match percentages for each category

pronouns = {
    1: [
        "I", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", 
        "I'm", "I've", "I'd", "I'll", "we're", "we've", "we'd", "we'll"
    ],
    2: [
        "you", "your", "yours", "yourself", "yourselves", "you're", "you've", "you'd", "you'll"
    ],
    3: [
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "one", "one's", "oneself",
        "he's", "she's", "it's", "they're", "they've", "he'd", "she'd", "it'd", "they'd",
        "he'll", "she'll", "it'll", "they'll"
    ]
}

def label_pronoun_with_percentages(text, categories):
    if isinstance(text, str):
        text = text.lower()  # Lowercase for case-insensitive matching
        category_count = {}
        total_count = 0

        # Count occurrences for each category
        for category, keywords in categories.items():
            count = sum(len(re.findall(r'\b' + re.escape(keyword) + r'\b', text)) for keyword in keywords)
            if count > 0:
                category_count[category] = count
                total_count += count

        # Calculate percentages if matches were found
        if total_count > 0:
            category_percentage = {category: (count / total_count) * 100 for category, count in category_count.items()}
            dominant_category = max(category_percentage, key=category_percentage.get)
            return dominant_category, category_percentage
    return 0, {}  # Default label if no matches

# Apply function to `Processed_Content` column and unpack results
df[['Perspective', 'Percentages']] = df['Title'].apply(
    lambda x: pd.Series(label_pronoun_with_percentages(x, pronouns))
)

df = df.iloc[:,:10]
df.head()


# In[39]:


# Categorizing army specific terms in titles, so general NLP models can understand it
army_categories = {
    'Medical': [
        'patient', 'hipaa', 'hippa', 'tricare', 'genesis', 'mhs', 'medical', 'doctor', 'hospital', 'mental health',
        'injury', 'appointment', 'bh', 'behavioral health', 'medpros', 'profile', 'sick call', 'chronic pain',
        'va', 'combat stress', 'post-traumatic stress', 'vaccination', 'dental', 'eye exam', 'pharmacy', 'ptsd'
    ],
    'Legal': [
        'court martial', 'court', 'gomor', 'gomar', 'legal', 'ucmj', 'jag', 'jags', 'tds', 'article 15', 'art 15', 
        'flag', 'administrative separation', 'bad conduct', 'discharge', 'court-martial', 'chapter', 'njp', 
        'field grade', 'general officer memorandum', 'restriction', 'reprimand', 'legal assistance', 'suspended'
    ],
    'Fitness': [
        'pt', 'fitness', 'apft', 'acft', 'running', 'spt', 'standing power throw', 'run', 'mile', 'ruck', 'deadlift',
        'pushups', 'push-up', 'two-mile run', 'leg tuck', 'sprint-drag-carry', 'plank', 'fitness test'
    ],
    'Substance Abuse': [
        'drinking', 'alcohol', 'drunk', 'beer', 'alcohol counseling', 'dui', 'alcohol-related incident', 'bar fight',
        'disorderly conduct', 'underage drinking', 'alcohol awareness', 'hangover', 'party', 'night out'
    ],
    'Deployment': [
        'deploying', 'deployment', 'deploy', 'redeploy', 'tour', 'overseas', 'combat zone', 'rotation', 'extended deployment',
        'pre-deployment', 'r&r', 'fob', 'oef', 'oif', 'homecoming', 'reunion', 'welcome home ceremony', 'oir'
    ]
}

generic_categories = {
    'HR': [
        'alaract', 'milper', 'correspondence', 'qmp', 'qsp', 'ribbon', 'brs', 'dod', 'policy', 'regulation',
        'business rules', 'dts', 'badge', 'badges', 'awards', 'awarded', 'vtip', 'ppw', 'award',
        'oer', 'ncoer', 'claims', 'personnel', 'promotion', 'admin', 'paperwork', 's1', 's-1', 
        'pax', 'clerk', 'leave', 'absence', 'ptdy', 'convalescent', 'ipps-a', 'ippsa',
        'maternity', 'paternity', 'caregiver', 'finance', 'deers', 'g1', 'hrc', 'orders', 'unit admin', 
        'leave form', 'travel claim', 'duty roster', 'accountability'
    ],
    'IT': [
        's6', 's-6', 'commo', 'cyber', 'network', 'communication equipment', 'computer', 
        'systems', 'software', 'communication', 'network security', 'account', 'hypori', 'remote desktop', 
        'signal', 'cyber awareness', 'AUP', '350-1'
    ],
    'Management': [
        'promoted', 'counsel', 'coach', 'mentor', 'counseling', 'pcs', 'pcsing', 'pcsd', 'branch manager', 'abcp',
        'leadership', 'commander', 'nco', 'career progression', 'cdr', 'bc', 'mentorship', 'development',
        'command climate', 'chain of command', 'platoon sergeant', 'first sergeant', 'company commander', 
        'squad leader', 'platoon leader', 'oic', 'ncoic', 'csm', '1sg', '1sgt', 'first sergeant', 
        'sergeant major', 'sgm', 'csm', 'sma', 'sergeant major of the army'
    ],
    'Customer': [
        'public service', 'community engagement', 'disaster relief', 'humanitarian aid', 'public relations',
        'support to civilians', 'veteran support', 'emergency response', 'citizen support', 'people of the United States'
    ],
    'New Hires': [
        'interservice', 'transfer', 'ship out', 'going army', 'enlisting', 'stationed', 'in basic', 'basic starts',
        'future soldier program', 'fsp', 'option 4', 'option 40', 'contract', 'asvab', 'ait', 'recruiting',
        'joining', 'enlistment', 'recruiter', 'meps', 'boot camp', 'mos', 'reclass', 'reclassing', 'basic training',
        'initial entry', 'signing bonus', 'drill sergeant', 'business rules'
    ],
    'Quality of Life': [
        'relationship', 'relationships', 'divorce', 'divorcing', 'divorced', 'children', 'family', 'spouse',
        'dependents', 'kids', 'childcare', 'family readiness', 'frg', 'military spouse employment', 'family housing',
        'family support', 'efmp', 'separation'
    ] + army_categories['Medical'] + army_categories['Legal'],
    'Product': [
        'soldier', 'service member', 'training', 'readiness', 'combat skills', 'development programs', 
        'military discipline', 'warfighter', 'military training', 'career development', 'physical fitness', 
        'deployment readiness', 'clearance', 's2', 's-2', 's3', 's-3', 's4', 's-4', 'tdy', 'promotion', 
        'finance', 'finances', 'financial readiness', 'deployability', 'deployment'
    ]  + army_categories['Fitness'],
    'Exit Survey': [
        'retention', 'reenlist', 'reenlistment', 're-enlist', 're-enlistment', 'reenlisted', 're-enlisted', 'sfl', 
        'soldier for life', 'ets', 'getting out', 'transition', 'compensation', 'benefits', 'uqr', 'refrad', 
        'separation', 'retirement', 'terminal leave', 'final pay', 'retirement ceremony', 'dd214', 'va benefits', 
        'disability claim', 'job search', 'civilian life', 'job placement'
    ] + army_categories['Substance Abuse'] + army_categories['Deployment']
}


# In[40]:


# Function to calculate keyword match percentages for each category
def label_title_with_percentages(text, categories):
    if isinstance(text, str):
        text = text.lower()  # Lowercase for case-insensitive matching
        category_count = {}
        total_count = 0

        # Count occurrences for each category
        for category, keywords in categories.items():
            count = sum(len(re.findall(r'\b' + re.escape(keyword) + r'\b', text)) for keyword in keywords)
            if count > 0:
                category_count[category] = count
                total_count += count

        # Calculate percentages if matches were found
        if total_count > 0:
            category_percentage = {category: (count / total_count) * 100 for category, count in category_count.items()}
            dominant_category = max(category_percentage, key=category_percentage.get)
            return dominant_category, category_percentage
    return 'Unlabeled', {}  # Default label if no matches

# Apply function to `Processed_Content` column and unpack results
df[['Category', 'Percentages']] = df['Title'].apply(
    lambda x: pd.Series(label_title_with_percentages(x, generic_categories))
)

df = df.iloc[:,:-1]
df


# In[41]:


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.special import softmax
from tqdm import tqdm
import pandas as pd

# Load the model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

df_test = df.iloc[:25,:]
titles = pd.read_excel("C:/Users/Sid/Desktop/OPER 655/Code/Preprocessing/Posts v2.xlsx", sheet_name = "Sheet2", header=None)
print(titles)
df_test['Title'] = titles
df_test


# In[42]:


# Define the sentiment analysis function
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    scores = outputs.logits[0].detach().numpy()
    scores = softmax(scores)

    # Get sentiment based on the highest score
    sentiments = ["Negative", "Neutral", "Positive"]
    sentiment = sentiments[scores.argmax()]
    confidence = float(scores.max())
    
    return sentiment, confidence

# Apply to each row in the DataFrame
tqdm.pandas()
df_test['Sentiment'], df_test['Confidence'] = zip(*df_test['Title'].progress_apply(analyze_sentiment))
df_test


# In[43]:


sentiment_map = {
                "Negative":-1,
                 "Neutral":0,
                 "Positive":1
                }
df_test["Sentiment"] = df_test["Sentiment"].map(sentiment_map)
df_test


# In[44]:


df_SA = df_test['Sentiment']
SA = np.array(df_SA.iloc[:25])
sentiments = np.array([1,1,1,1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,0,1,0,-1,-1,1,0,-1,-1,-1])


# In[45]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

labels = [-1, 0, 1]

# Generate the confusion matrix
cm = confusion_matrix(sentiments, SA, labels=labels)

# Create the confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Neutral", "Positive"])
disp.plot(colorbar=False)


# In[46]:


print(classification_report(sentiments,SA))


# In[47]:


incorrect_indices = np.where(SA != sentiments)[0]

# Create a DataFrame with the mismatched indices and corresponding values from both arrays
confidence = [1, 0.778, 0.778, 0.667, 0.556, 0.556, 0.556, 0.444, 0.778, 0.444]
incorrect_df = pd.DataFrame({
    'Question number':incorrect_indices+1,
    'Predicted': SA[incorrect_indices],
    'Actual': sentiments[incorrect_indices],
    'Human Confidence': confidence
})

incorrect_df



# ## 60% accuracy; however, when adjusting for mistakes of only overwhelming majority response (>=2/3), accuracy is 80%

# In[49]:


df['Sentiment'], df['Confidence'] = zip(*df['Title'].progress_apply(analyze_sentiment))                
df["Sentiment"] = df["Sentiment"].map(sentiment_map)
df


# In[50]:


def has_caps(title):
    # Use regex to find all words in the title
    words = re.findall(r'\b[A-Z]+\b', title)
    return 1 if len(words) > 0 else 0

df["Caps"] = df["Title"].apply(has_caps)
df


# In[51]:


import emoji

emoji_list = list(emoji.EMOJI_DATA.keys())


# In[52]:


abbreviations = {
    " $ " : " dollar ",
    " € " : " euro ",
    " 4ao " : "for adults only",
    " a3 " : "anytime anywhere anyplace",
    " aamof " : "as a matter of fact",
    " acct " : "account",
    " adih " : "another day in hell",
    " afaic " : "as far as i am concerned",
    " afaict " : "as far as i can tell",
    " afaik " : "as far as i know",
    " afk " : "away from keyboard",
    " app " : "application",
    " approx " : "approximately",
    " apps " : "applications",
    " asap " : "as soon as possible",
    " asl " : "age, sex, location",
    " atk " : "at the keyboard",
    " aymm " : "are you my mother",
    " ayor " : "at your own risk", 
    " b&b " : "bed and breakfast",
    " b+b " : "bed and breakfast",
    " b2b " : "business to business",
    " b2c " : "business to customer",
    " b4 " : "before",
    " b4n " : "bye for now",
    " b@u " : "back at you",
    " bae " : "before anyone else",
    " bak " : "back at keyboard",
    " bbbg " : "bye bye be good",
    " bbc " : "british broadcasting corporation",
    " bbias " : "be back in a second",
    " bbl " : "be back later",
    " bbs " : "be back soon",
    " be4 " : "before",
    " bfn " : "bye for now",
    " bout " : "about",
    " brb " : "be right back",
    " bros " : "brothers",
    " brt " : "be right there",
    " bsaaw " : "big smile and a wink",
    " btw " : "by the way",
    " bwl " : "bursting with laughter",
    " c/o " : "care of",
    " cf " : "compare",
    " csl " : "can not stop laughing",
    " cu " : "see you",
    " cul8r " : "see you later",
    " cwot " : "complete waste of time",
    " cya " : "see you",
    " cyt " : "see you tomorrow",
    " dae " : "does anyone else",
    " dbmib " : "do not bother me i am busy",
    " diy " : "do it yourself",
    " dm " : "direct message",
    " dwh " : "during work hours",
    " e123 " : "easy as one two three",
    " faq " : "frequently asked questions",
    " fawc " : "for anyone who cares",
    " fb " : "facebook",
    " fc " : "fingers crossed",
    " fig " : "figure",
    " fimh " : "forever in my heart", 
    " ftl " : "for the loss",
    " ftw " : "for the win",
    " fwiw " : "for what it is worth",
    " fyi " : "for your information",
    " g9" : "genius",
    " gahoy " : "get a hold of yourself",
    " gal " : "get a life",
    " gcse " : "general certificate of secondary education",
    " gfn " : "gone for now",
    " gg " : "good game",
    " gl " : "good luck",
    " glhf " : "good luck have fun",
    " gmt " : "greenwich mean time",
    " gmta " : "great minds think alike",
    " gn " : "good night",
    " g.o.a.t " : "greatest of all time",
    " goat " : "greatest of all time",
    " goi " : "get over it",
    " gr8 " : "great",
    " gratz " : "congratulations",
    " gyal " : "girl",
    " h&c " : "hot and cold",
    " hrh " : "his royal highness",
    " ibrb " : "i will be right back",
    " ic " : "i see",
    " icq " : "i seek you",
    " icymi " : "in case you missed it",
    " idc " : "i do not care",
    " idgadf " : "i do not give a damn fuck",
    " idgaf " : "i do not give a fuck",
    " idk " : "i do not know",
    " ifyp " : "i feel your pain",
    " IG " : "instagram",
    " iirc " : "if i remember correctly",
    " ilu " : "i love you",
    " ily " : "i love you",
    " imho " : "in my humble opinion",
    " imo " : "in my opinion",
    " imu " : "i miss you",
    " iow " : "in other words",
    " irl " : "in real life",
    " j4f " : "just for fun",
    " jic " : "just in case",
    " jk " : "just kidding",
    " jsyk " : "just so you know",
    " l8r " : "later",
    " ldr " : "long distance relationship",
    " lmao " : "laugh my ass off",
    " lmfao " : "laugh my fucking ass off",
    " lol " : "laughing out loud",
    " ltd " : "limited",
    " ltns " : "long time no see",
    " m8 " : "mate",
    " mf " : "motherfucker",
    " mfs " : "motherfuckers",
    " mfw " : "my face when",
    " mofo " : "motherfucker",
    " mrw " : "my reaction when",
    " mte " : "my thoughts exactly",
    " nagi " : "not a good idea",
    " nbd " : "not big deal",
    " nfs " : "not for sale",
    " ngl " : "not going to lie",
    " nrn " : "no reply necessary",
    " nsfl " : "not safe for life",
    " nsfw " : "not safe for work",
    " nth " : "nice to have",
    " nvr " : "never",
    " oc " : "original content",
    " og " : "original",
    " oic " : "oh i see",
    " omdb " : "over my dead body",
    " omg " : "oh my god",
    " omw " : "on my way",
    " poc " : "people of color",
    " pov " : "point of view",
    " pp " : "pages",
    " ppl " : "people",
    " prw " : "parents are watching",
    " ptb " : "please text back",
    " pto " : "please turn over",
    " ratchet " : "rude",
    " rbtl " : "read between the lines",
    " rlrt " : "real life retweet", 
    " rofl " : "rolling on the floor laughing",
    " roflol " : "rolling on the floor laughing out loud",
    " rotflmao " : "rolling on the floor laughing my ass off",
    " rt " : "retweet",
    " ruok " : "are you ok",
    " sfw " : "safe for work",
    " sk8 " : "skate",
    " smh " : "shake my head",
    " sq " : "square",
    " srsly " : "seriously", 
    " ssdd " : "same stuff different day",
    " tbh " : "to be honest",
    " tfw " : "that feeling when",
    " thks " : "thank you",
    " tho " : "though",
    " thx " : "thank you",
    " tia " : "thanks in advance",
    " til " : "today i learned",
    " tl;dr " : "too long i did not read",
    " tldr " : "too long i did not read",
    " tmb " : "tweet me back",
    " tntl " : "trying not to laugh",
    " ttyl " : "talk to you later",
    " u " : "you",
    " u2 " : "you too",
    " u4e " : "yours for ever",
    " w/ " : "with",
    " w/o " : "without",
    " w8 " : "wait",
    " wassup " : "what is up",
    " wb " : "welcome back",
    " wtf " : "what the fuck",
    " wtg " : "way to go",
    " wtpa " : "where the party at",
    " wuf " : "where are you from",
    " wuzup" : "what is up",
    " wywh " : "wish you were here",
    " ygtr " : "you got that right",
    " ynk " : "you never know",
    " zzz " : "sleeping bored and tired"
}


# from https://www.kaggle.com/code/nmaguette/up-to-date-list-of-slangs-for-text-preprocessing

# In[54]:


slang_list = list(abbreviations.keys())
slang_list


# In[55]:


def contains_slang(title, slang_list):
    if any(slang in title.lower() for slang in slang_list):
        return 1  
    else:
        return 0  

def contains_emoji(title, emoji_list):
    if any(emoji in title.lower() for emoji in emoji_list):
        return 1  
    else:
        return 0  

df['Slang'] = df['Title'].apply(lambda x: contains_slang(x, slang_list))
df['Emoji'] = df['Title'].apply(lambda x: contains_emoji(x, emoji_list))
df


# In[56]:


value_counts = df['Emoji'].value_counts()
value_counts


# In[57]:


value_counts = df['Slang'].value_counts()
value_counts


# In[58]:


# Function to count POS tags
def count_pos_tags(title):
    doc = nlp(title)
    pos_counts = Counter([token.pos_ for token in doc])  # Count each POS tag
    return pos_counts

# Apply POS counting function and expand results into separate columns
pos_counts_df = df['Title'].apply(count_pos_tags).apply(pd.Series).fillna(0)

# Concatenate the original DataFrame with the new POS counts DataFrame
df1 = pd.concat([df, pos_counts_df], axis=1)


# In[59]:


print(df1.columns)


# ### Part of Speech Legend
# 
# | POS Tag  | Description                                | Example                  |
# |----------|--------------------------------------------|--------------------------|
# | PRON     | Pronoun                                   | he, she, they, it       |
# | PART     | Particle                                  | to (in "to run"), not   |
# | VERB     | Verb                                      | run, is, write          |
# | ADP      | Adposition (preposition or postposition)  | on, in, under           |
# | PROPN    | Proper noun                               | John, America, Google   |
# | PUNCT    | Punctuation                               | ., !, ?                 |
# | ADJ      | Adjective                                 | happy, blue, fast       |
# | NOUN     | Noun                                      | dog, city, happiness    |
# | AUX      | Auxiliary verb                            | is, has, do             |
# | SCONJ    | Subordinating conjunction                | because, if, although   |
# | DET      | Determiner                                | the, a, an              |
# | CCONJ    | Coordinating conjunction                 | and, or, but            |
# | NUM      | Numeral                                   | one, 2, first           |
# | ADV      | Adverb                                    | quickly, very, well     |
# | SYM      | Symbol                                    | $, %, +                 |
# | INTJ     | Interjection                              | wow, hey, ouch          |
# | X        | Other (unclassified or typo)             | (varies)                |
# | SPACE    | Space (whitespace between words)         | (whitespace character)  |
# 
# 

# In[61]:


value_counts = df['Season posted'].value_counts()
value_counts


# In[62]:


value_counts = df['Time of Day'].value_counts()
value_counts


# In[63]:


value_counts = df['Sentiment'].value_counts()
value_counts


# In[64]:


value_counts = df['Question'].value_counts()
value_counts


# In[65]:


value_counts = df['Quote'].value_counts()
value_counts


# In[66]:


value_counts = df['Caps'].value_counts()
value_counts


# In[67]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['Upvotes'] = scaler.fit_transform(df['Upvotes'].values.reshape(-1, 1))
df['Number of Comments'] = scaler.fit_transform(df['Number of Comments'].values.reshape(-1, 1))


# In[68]:


df['Engagement'] = 0.1*df['Number of Comments']+0.9*df['Upvotes']
df.head()


# In[69]:


def engagement_map(engagement_score):
    if engagement_score > 0:
        return 1
    else:
        return 0

df['Engagement'] = df['Engagement'].apply(engagement_map)
df


# In[70]:


df1['Engagement'] = df['Engagement']


# In[71]:


df = df.drop(columns = ["Slang", "Emoji", "Confidence", "Upvotes", "Number of Comments"])


# In[72]:


df.to_excel("Cleaned data wo pos.xlsx", index = False)


# In[73]:


df1 = df1.drop(columns = ["Slang", "Emoji", "Confidence", "Upvotes", "Number of Comments"])


# In[74]:


df1.to_excel("Cleaned data w pos.xlsx", index = False)


# In[ ]:




