#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, classification_report, auc, accuracy_score


# In[5]:


titles = pd.read_csv("Titles.csv")
df_tf = pd.read_csv("TF-IDF.csv")
df_tf


# In[6]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
columns_to_scale = ['Title Length', 'Title Word Count']
df_tf[columns_to_scale] = scaler.fit_transform(df_tf[columns_to_scale])


# In[7]:


X = df_tf.drop(columns = ['Engagement'])
y = df_tf['Engagement']
X


# In[8]:


print(np.round(y.value_counts(normalize=True) * 100,3))


# In[9]:


# Split data into training (70%), validation (15%), and test (15%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# In[7]:


print(X_train.shape)


# # Logisitc Regression

# In[89]:


from sklearn.linear_model import LogisticRegression

model_log = LogisticRegression()


# In[91]:


param_grid = {
    'C': [0.1, 1, 10, 100],  # higher means more likely to overfit
    'penalty': ['l1', 'l2', None],  # None for no penalty
    'solver': ['saga', 'liblinear', 'lbfgs', 'newton-cg'],
    'class_weight': ['balanced']
}

# Define the logistic regression model
model_log = LogisticRegression()

# Create GridSearchCV with the updated param_grid
grid_search = GridSearchCV(estimator=model_log, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the grid search on your training data
grid_search.fit(X_train, y_train)


# In[92]:


# Get the best hyperparameters
best_params = grid_search.best_params_

print("Best hyperparameters:", best_params)


# In[93]:


best_model = grid_search.best_estimator_

y_val_pred = best_model.predict(X_valid)


# In[94]:


# Calculate accuracy on the validation set
val_accuracy = accuracy_score(y_valid, y_val_pred)
print(f"Validation Accuracy: {val_accuracy*100}%")


# In[95]:


y_pred = best_model.predict(X_test)
labels = {0:'Not Engaging',
          1:'Engaging'}
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix= cm,display_labels=[labels[i] for i in range(len(labels))])
disp.plot(colorbar = False)

plt.title('Confusion Matrix for Engagement predictions')


# In[96]:


report  = classification_report(y_test, y_pred)
print(report)


# In[66]:


# DiCE imports
import dice_ml
from dice_ml.utils import helpers  # helper functions


# In[16]:


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


# In[23]:


m = dice_ml.Model(model=best_model, backend="sklearn")
exp = dice_ml.Dice(d, m, method="random")


# In[24]:


X_train_sample = X_train.sample(10, random_state = 42)
cobj = exp.global_feature_importance(X_train_sample)


# In[25]:


print(cobj.summary_importance)


# In[97]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[27]:


dice_log = {key: value * accuracy for key, value in cobj.summary_importance.items()}
dice_log


# In[ ]:


from shapash import SmartExplainer


# In[68]:


xpl = SmartExplainer(model = best_model)


# In[69]:


ypred = pd.DataFrame(y_pred)
ypred


# In[92]:


Xtest = X_test.reset_index(drop = True)
ytest = y_test.reset_index(drop = True)


# In[94]:


xpl.compile(x=Xtest,
 y_pred=ypred,
 y_target=ytest, # Optional: allows to display True Values vs Predicted Values
 )


# In[98]:


xpl.plot.features_importance(mode = 'global', max_features = 10)


# In[ ]:


xpl.plot.contribution_plot("Sec")


# ## Shap

# In[98]:


import shap

# Assuming you already have a trained model and a dataset (X_train)
explainer = shap.Explainer(best_model)  # Replace with your model
shap_values = explainer.shap_values(X_train)


# In[ ]:





# # Neural Network

# In[19]:


n = X_train.shape[1]
model = Sequential()

model.add(Dense(64, activation='relu', use_bias=False, bias_initializer='ones', input_shape = (n,))) # -
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

opt = tf.keras.optimizers.Adam(learning_rate=5e-6)  
model.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss', 
    min_delta = 0.001,
    patience=10,    
    restore_best_weights=True 
)

history = model.fit(X_train, y_train, epochs = 50, validation_data = (X_valid, y_valid),
                    batch_size=1, verbose = 1, callbacks = [early_stopping]) 


# In[20]:


loss, mse = model.evaluate(X_test, y_test)
print(f"Model mse: {mse}")
print(f"Model loss: {loss}")


# In[21]:


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title("Train and Validation Metric Values")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()


# In[22]:


preds = model.predict(X_test)
y_pred = (preds >= (0.35)).astype(int)


# In[23]:


labels = {0:'Not Engaging',
          1:'Engaging'}
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix= cm,display_labels=[labels[i] for i in range(len(labels))])
disp.plot(colorbar = False)

plt.title('Confusion Matrix for Engagement predictions')


# In[24]:


report  = classification_report(y_test, y_pred)
print(report)


# In[73]:


model.save('engagement_NN.h5')


# In[83]:


model_path = "C:/Users/Sid/Desktop/OPER 655/Code/Model/engagement_NN.h5"
m = dice_ml.Model(model_path=model_path, backend="TF2", func="ohe-min-max")
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


# In[85]:


exp = dice_ml.Dice(d, m, method="gradient")


# In[ ]:


cobj = exp.global_feature_importance(X_train_sample)


# In[ ]:


print(cobj.summary_importance)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[ ]:


dice_NN = {key: value * accuracy for key, value in cobj.summary_importance.items()}
dice_NN


# ## Shapash

# In[25]:


xpl = SmartExplainer(model = model)


# In[26]:


ypred = pd.DataFrame(y_pred)


# In[27]:


Xtest = X_test.reset_index(drop = True)
ytest = y_test.reset_index(drop = True)


# In[28]:


xpl.compile(x=Xtest,
 y_pred=ypred,
 y_target=ytest, # Optional: allows to display True Values vs Predicted Values
 )


# In[29]:


xpl.plot.features_importance(mode = 'global', max_features = 10)


# # Random Forest

# In[16]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights based on imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)

# Set up hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Perform Grid Search for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


# In[ ]:


best_params = grid_search.best_params_

print("Best hyperparameters:", best_params)


# In[ ]:


best_rf_model = grid_search.best_estimator_

y_pred = best_rf_model.predict(X_test)


# In[ ]:


labels = {0:'Not Engaging',
          1:'Engaging'}
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix= cm,display_labels=[labels[i] for i in range(len(labels))])
disp.plot(colorbar = False)

plt.title('Confusion Matrix for Engagement predictions')


# In[ ]:


report  = classification_report(y_test, y_pred)
print(report)


# In[20]:


m = dice_ml.Model(model=best_rf_model, backend="sklearn")
exp = dice_ml.Dice(d, m, method="random")


# In[34]:


cobj = exp.global_feature_importance(X_train_sample)


# In[35]:


print(cobj.summary_importance)


# In[54]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[37]:


dice_rf = {key: value * accuracy for key, value in cobj.summary_importance.items()}
dice_rf


# ## Shapash

# In[ ]:


xpl = SmartExplainer(model = best_rf_model)


# In[ ]:


ypred = pd.DataFrame(y_pred)


# In[ ]:


Xtest = X_test.reset_index(drop = True)
ytest = y_test.reset_index(drop = True)


# In[ ]:


xpl.compile(x=Xtest,
 y_pred=ypred,
 y_target=ytest, # Optional: allows to display True Values vs Predicted Values
 )


# In[ ]:


xpl.plot.features_importance(mode = 'global', max_features = 10)


# In[ ]:


xpl.plot.contribution_plot("Season posted")


# In[26]:


import shap

# Assuming you already have a trained model and a dataset (X_train)
explainer = shap.TreeExplainer(best_rf_model)  # Replace with your model
shap_values = explainer.shap_values(X_train)


# In[42]:


# Get the absolute mean of SHAP values for feature importance
feature_importance = np.abs(shap_values).mean(axis=0)
scores = feature_importance[:,0]


# In[56]:


# Combine with feature names
feature_importance = dict(zip(X_train.columns, scores))
shap_rf = {key: value * accuracy for key, value in feature_importance.items()}
shap_rf


# In[46]:


# Sort by importance
sorted_feature_importance = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))


# In[48]:


# Plot top 10 features
top_features = list(sorted_feature_importance.keys())[:10]
top_importance = list(sorted_feature_importance.values())[:10]

plt.barh(top_features[::-1], top_importance[::-1])
plt.xlabel("Mean Absolute SHAP Value")
plt.title("Feature Importance")


# ## Permutation

# In[113]:


# Get feature importances
feature_importances = best_rf_model.feature_importances_
feature_names = X_train.columns

# Create a pandas Series
mdi_importances = pd.Series(feature_importances, index=feature_names).sort_values(ascending=True)


# In[119]:


top_10_importances = mdi_importances.nlargest(10)

top_10_importances = top_10_importances.sort_values(ascending=True)


# In[121]:


top_10_importances.plot(kind='barh', color='skyblue')
plt.title("Top 10 Feature Importances (MDI)")
plt.xlabel("Importance")
plt.tight_layout()


# # SVM

# In[61]:


from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Initialize GridSearchCV with SVM and the parameter grid
grid_search = GridSearchCV(SVC(probability = True, random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Find the best estimator
best_svm_model = grid_search.best_estimator_


# In[62]:


best_params = grid_search.best_params_

print("Best hyperparameters:", best_params)


# In[63]:


best_svm_model = grid_search.best_estimator_

y_pred = best_svm_model.predict(X_test)


# In[64]:


labels = {0:'Not Engaging',
          1:'Engaging'}
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix= cm,display_labels=[labels[i] for i in range(len(labels))])
disp.plot(colorbar = False)

plt.title('Confusion Matrix for Engagement predictions')


# In[65]:


report  = classification_report(y_test, y_pred)
print(report)


# In[77]:


m = dice_ml.Model(model=best_svm_model, backend="sklearn", func="ohe-min-max")
exp = dice_ml.Dice(d, m, method="random")


# In[56]:


cobj = exp.global_feature_importance(X_train_sample)


# In[ ]:


print(cobj.summary_importance)


# In[66]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[ ]:


dice_svm = {key: value * accuracy for key, value in cobj.summary_importance.items()}
dice_svm


# ## Shapash

# In[92]:


xpl = SmartExplainer(model = best_svm_model)


# In[94]:


ypred = pd.DataFrame(y_pred)


# In[96]:


xpl.compile(x=Xtest,
 y_pred=ypred,
 y_target=ytest, # Optional: allows to display True Values vs Predicted Values
 )


# In[97]:


xpl.plot.features_importance(mode = 'global', max_features = 10)


# ## Shap

# In[67]:


import shap

# Assuming you already have a trained model and a dataset (X_train)
explainer = shap.Explainer(best_svm_model)  # Replace with your model
shap_values = explainer.shap_values(X_train)


# In[ ]:


# Get the absolute mean of SHAP values for feature importance
feature_importance = np.abs(shap_values).mean(axis=0)
scores = feature_importance[:,0]


# In[ ]:


# Combine with feature names
feature_importance = dict(zip(X_train.columns, scores))
shap_svm = {key: value * accuracy for key, value in feature_importance.items()}
shap_svm


# # CNN

# In[5]:


df_bert = pd.read_csv("Bert.csv")
print(df_bert.shape)


# In[8]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
columns_to_scale = ['Title Length', 'Title Word Count']
df_bert[columns_to_scale] = scaler.fit_transform(df_bert[columns_to_scale])


# In[10]:


y = df_bert['Engagement']


# In[14]:


features = np.array(df_bert.drop(columns = ["Title_embeddings", "Engagement"]))
print(features.shape)
embeddings = np.array([np.fromstring(str(embedding).strip('[]'), sep=' ') for embedding in df_bert['Title_embeddings'].tolist()])
print(embeddings.shape)


# In[16]:


def data_3d_cross(df1,df2):
    i = df1.shape[0]
    j = df1.shape[1]
    k = df2.shape[1]
    result = np.empty((i,j,k), dtype = int)

    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            result[i,j] = [df1[i,j]] + list(df2[i, :k-1])

    return result

data_3d = data_3d_cross(features,embeddings)
print(data_3d.shape)


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(data_3d, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# In[69]:


from tensorflow.keras import models, layers, optimizers

model = models.Sequential()

# First Convolutional Layer
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(27,768,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))

# Second Convolutional Layer
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

# Flattening the layers
model.add(layers.Flatten())
model.add(layers.Dropout(0.25))

# First Dense layer
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.25))

# Second Dense layer
model.add(layers.Dense(16, activation='relu'))

# Output layer
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()


# In[77]:


opt = optimizers.Adam(learning_rate=0.00005)
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

es = EarlyStopping(monitor = 'val_loss', mode = 'min', min_delta = 0.01, patience = 10)

history = model.fit(X_train, y_train, batch_size=64, epochs=25,
                     validation_data=(X_valid, y_valid),
                     callbacks = [es], verbose = 2)

# Test the model's accuracy with the test data
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)


# In[78]:


# Plot only the loss values
pd.DataFrame({
    "train_loss": history.history["loss"],
    "val_loss": history.history["val_loss"]
}).plot(figsize=(8, 5))

plt.grid(True)
plt.gca().set_ylim(0, 1)  # Adjust y-axis limits as needed
plt.title("Train and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[79]:


y_pred_probs = model.predict(X_test)

y_pred = (y_pred_probs > 0.5).astype("int32")

labels = {0:'Not Engaging',
          1:'Engaging'}
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix= cm)
disp.plot(colorbar = False)


# In[80]:


report  = classification_report(y_test, y_pred)
print(report)


# In[35]:


plt.hist(y_pred_probs, bins = 20)
plt.title("Sigmoid Output Distribution")
plt.xlabel("Output")
plt.ylabel("Frequency")


# ## Saving best model

# In[ ]:


model.save('engagement_classifier.h5')
np.save("X_test.npy",X_test)
np.save("y_test.npy",y_test)
np.save("y_pred.npy",y_pred)


# # Total feature importance

# In[88]:


total_dice = {key: (dice_log[key] + dice_rf[key]) / 2 for key in dice_rf.keys()}
total_dice


# In[104]:


sorted_data = dict(sorted(total_dice.items(), key=lambda item: item[1]))

top_10 = dict(list(sorted_data.items())[-10:])

keys = list(top_10.keys())
values = list(top_10.values())

# Create the horizontal bar chart
plt.barh(keys, values, color='purple')

# Add labels and title
plt.xlabel('Global Importance')
plt.ylabel('Feature')
plt.title("10 Most Important Features")


# In[ ]:


total_shap = {key: (shap_log[key] + shap_rf[key] + shap_svm[key] + shap_NN[key]) / 4 for key in shap_rf.keys()}
total_shap


# In[ ]:


sorted_data = dict(sorted(total_dice.items(), key=lambda item: item[1]))

top_10 = dict(list(sorted_data.items())[-10:])

keys = list(top_10.keys())
values = list(top_10.values())

# Create the horizontal bar chart
plt.barh(keys, values, color='yellow')

# Add labels and title
plt.xlabel('Shap Global Importance')
plt.ylabel('Feature')
plt.title("10 Most Important Features")


# # LSTM

# In[46]:


model = models.Sequential()

# First LSTM Layer
model.add(layers.LSTM(128, activation='relu', return_sequences=True, input_shape=(28, 768)))
model.add(layers.Dropout(0.5))

# Second LSTM Layer
model.add(layers.LSTM(64, activation='relu', return_sequences=False))
model.add(layers.Dropout(0.2))  

# First Dense Layer
model.add(layers.Dense(16, activation='relu'))

# Second Dense Layer
model.add(layers.Dense(8, activation='relu'))

# Output Layer
model.add(layers.Dense(1, activation='sigmoid')) 


# In[50]:


opt = optimizers.Adam(learning_rate=0.00005)
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

es = EarlyStopping(monitor = 'val_loss', mode = 'min', min_delta = 0.01, patience = 10)

history = model.fit(X_train, y_train, batch_size=64, epochs=50,
                     validation_data=(X_valid, y_valid),
                     callbacks = [es], verbose = 2)

# Test the model's accuracy with the test data
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)


# In[66]:


# Plot only the loss values
pd.DataFrame({
    "train_loss": history.history["loss"],
    "val_loss": history.history["val_loss"]
}).plot(figsize=(8, 5))

plt.grid(True)
plt.gca().set_ylim(0, 1)  # Adjust y-axis limits as needed
plt.title("Train and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[62]:


y_pred_probs = model.predict(X_test)

y_pred = (y_pred_probs > 0.5).astype("int32")

labels = {0:'Not Engaging',
          1:'Engaging'}
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix= cm)
disp.plot(colorbar = False)


# In[64]:


report  = classification_report(y_test, y_pred)
print(report)


# In[ ]:




