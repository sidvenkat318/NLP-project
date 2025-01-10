#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import dice_ml
from shapash import SmartExplainer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score


# # Preprocessing

# In[3]:


df_tf = pd.read_csv("TF-IDF.csv")


# In[4]:


scaler = StandardScaler()
columns_to_scale = ['Title Length', 'Title Word Count']
df_tf[columns_to_scale] = scaler.fit_transform(df_tf[columns_to_scale])


# In[5]:


X = df_tf.drop(columns = ['Engagement'])
y = df_tf['Engagement']


# In[6]:


# Split data into training (70%), validation (15%), and test (15%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# # Random Forest

# In[13]:


# Calculate class weights based on imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

custom_params = {
    'n_estimators': 200,
    'max_depth': 20,
    'max_features': 'log2',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'bootstrap': True,
    'random_state': 42,
    'class_weight': class_weight_dict
}

# Create the Random Forest model with your parameters
rf = RandomForestClassifier(**custom_params)
rf.fit(X_train, y_train)


# In[15]:


y_pred = rf.predict(X_test)

labels = {0:'Not Engaging',
          1:'Engaging'}
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix= cm,display_labels=[labels[i] for i in range(len(labels))])
disp.plot(colorbar = False)

plt.title('Confusion Matrix for Engagement predictions')


# In[17]:


report  = classification_report(y_test, y_pred)
print(report)


# # Shapash

# In[20]:


xpl = SmartExplainer(model = rf)


# In[22]:


ypred = pd.DataFrame(y_pred)


# In[24]:


Xtest = X_test.reset_index(drop = True)
ytest = y_test.reset_index(drop = True)


# In[26]:


xpl.compile(x=Xtest,
           y_target = ytest)


# In[27]:


app = xpl.run_app(title_story='Engagement Classifier')


# In[28]:


app.kill()


# # MDI

# In[28]:


# Get feature importances
feature_importances = rf.feature_importances_
feature_names = X_train.columns

# Create a pandas Series
mdi_importances = pd.Series(feature_importances, index=feature_names).sort_values(ascending=True)


# In[29]:


top_10_importances = mdi_importances.nlargest(10)

top_10_importances = top_10_importances.sort_values(ascending=True)


# In[30]:


top_10_importances.plot(kind='barh', color='skyblue')
plt.title("Top 10 Feature Importances (MDI)")
plt.xlabel("Importance")
plt.tight_layout()


# # DiCE

# In[73]:


m = dice_ml.Model(model=rf, backend="sklearn")
d = dice_ml.Data(
    dataframe=df_tf, 
    continuous_features=["about", "all", "an", "and", "are", "army", "as", "at", "be", "can", "do", "for", "fort", "from", "get",
                         "got", "have", "help", "how", "if", "in", "is", "it", "just", "like", "me", "military", "my", "new", "not",
                         "of", "on", "one", "or", "out", "question", "soldiers", "that", "the", "this", "to", "us", "was", "we", "what",
                         "who",	"why", "with", "you", "your", "Title Length", "Title Word Count", 
                         "PRON", "PART", "VERB", "ADP", "PROPN", "PUNCT", "ADJ", "NOUN", "AUX", "SCONJ", "DET", "CCONJ", "NUM", "ADV", 
                         "SYM", "INTJ", "X", "SPACE"],
    categorical_features=["Season posted", "Time of Day", "Sentiment", "Question", "Quote", "Category", "Caps"], 
    outcome_name='Engagement'             
)
exp = dice_ml.Dice(d, m, method="random")


# In[75]:


X_train_sample = X_train.sample(10, random_state = 42)
cobj = exp.global_feature_importance(X_train_sample)


# In[78]:


dir(cobj)


# In[84]:


sorted_data = dict(sorted(cobj.summary_importance.items(), key=lambda item: item[1]))

top_10 = dict(list(sorted_data.items())[-10:])

keys = list(top_10.keys())
values = list(top_10.values())

# Create the horizontal bar chart
plt.barh(keys, values, color='purple')

# Add labels and title
plt.xlabel('Global Importance')
plt.ylabel('Feature')
plt.title("10 Most Important Features")


# # Permutation

# In[97]:


from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)

sorted_importances_idx = result.importances_mean.argsort()[::-1][:10]
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X.columns[sorted_importances_idx],
)

# Reverse the column order so the most important is at the top
importances = importances.loc[:, ::-1]

ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()


# In[ ]:




