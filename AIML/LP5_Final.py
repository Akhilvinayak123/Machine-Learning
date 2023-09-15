#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


pd.options.mode.chained_assignment = None


# In[3]:


lf = pd.read_excel("LP5.xlsx")


# In[4]:


lf


# In[5]:


lfnew = lf[["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]].copy()


# In[6]:


lf2=lfnew.dropna()


# In[7]:


lf2


# In[8]:


lf2 = lf2.reset_index(drop = True)


# In[9]:


lf3 = lf2.groupby(np.arange(len(lf2))//15).mean()


# In[10]:


lf3


# In[11]:


lf5 = pd.read_excel("LP5types.xlsx")


# In[12]:


lf5


# In[13]:


lf5 = lf5.dropna(axis = 0)


# In[14]:


lf5 = lf5.reset_index(drop = True)


# In[15]:


lf5.head(10)


# In[16]:


lf3["Failure_type"] = lf5


# In[17]:


lf3


# In[18]:


lf3.to_csv('LP5S.csv')


# In[19]:



sns.countplot(x = "Failure_type", data = lf3);


# In[20]:


fig, axes = plt.subplots(6)
sns.boxplot(x = "Fx", data = lf3, ax =axes[0]);
sns.boxplot(x = "Fy", data = lf3, ax =axes[1]);
sns.boxplot(x = "Fz", data = lf3, ax =axes[2]);
sns.boxplot(x = "Tx", data = lf3, ax =axes[3]);
sns.boxplot(x = "Ty", data = lf3, ax =axes[4]);
sns.boxplot(x = "Tz", data = lf3, ax =axes[5]);



# # Dataset

# In[21]:


lf3


# # Label encoder

# In[22]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
lf3.Failure_type = le.fit_transform(lf3.Failure_type)
lf3.head()


# # Data split

# In[23]:


from sklearn.model_selection import train_test_split
X = lf3.drop(['Failure_type'], axis = 1)
Y = lf3['Failure_type']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size =0.33,  random_state = 2)


# In[24]:


X_train.head()


# In[25]:


X_test.head()


# In[26]:


Y_train.head()


# In[27]:


Y_test.head()


# # Decision Tree

# In[28]:


get_ipython().run_cell_magic('time', '', "#Decision Tree\n\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import confusion_matrix\nclf = DecisionTreeClassifier(criterion='gini', max_depth = 5, splitter='best')\nclf = clf.fit(X_train, Y_train)")


# In[29]:


clf.get_params()


# In[30]:


X_test;


# In[31]:


predictions = clf.predict(X_test)
predictions


# In[32]:


clf.predict_proba(X_test);


# In[33]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,predictions)


# In[34]:


c = confusion_matrix(Y_test, predictions)
print(c)
sns.heatmap(c, annot=True, fmt=".1f")
plt.xlabel('Actual');
plt.ylabel('Predicted');


# In[35]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))


# In[36]:


from sklearn import tree
plt.figure(figsize = (8,6))
tree.plot_tree(clf,filled = True, class_names = True);


# # KNN
# 

# In[37]:


get_ipython().run_cell_magic('time', '', "from sklearn.neighbors import KNeighborsClassifier\n\nneigh = KNeighborsClassifier(n_neighbors=3,weights='distance',leaf_size=30)\nneigh.fit(X_train, Y_train)")


# In[38]:


Y_pred = neigh.predict(X_test)
print(Y_pred)


# In[39]:


neigh.predict_proba(X_test)


# In[40]:


c = confusion_matrix(Y_test, Y_pred)
print(c)
sns.heatmap(c, annot=True, fmt=".1f")
plt.xlabel('Actual');
plt.ylabel('Predicted');


# In[41]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))


# In[42]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred)


# # Randomforest
# 

# In[43]:


from sklearn.ensemble import RandomForestClassifier
  


# In[44]:


get_ipython().run_cell_magic('time', '', 'rfc = RandomForestClassifier(n_estimators=500, max_depth=4, random_state = 2)\nrfc.fit(X_train, Y_train)')


# In[45]:


Y_predicts = rfc.predict(X_test)
print(Y_predicts)


# In[46]:


rfc.predict_proba(X_test);


# In[47]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_predicts))


# In[48]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_predicts)


# In[49]:


c = confusion_matrix(Y_test, Y_predicts)
print(c)
sns.heatmap(c, annot=True, fmt=".1f")
plt.xlabel('Actual');
plt.ylabel('Predicted');


# In[ ]:





# # SVM

# In[50]:


#%%time
#from sklearn.svm import SVC

#model = SVC(kernel="poly", degree=8, gamma = 'auto')

#model.fit(X_train,Y_train)
#Y_predict = model.predict(X_test)


# In[51]:


#print(classification_report(Y_test, Y_predict))


# In[52]:


#accuracy_score(Y_test,Y_predict)


# In[53]:


#c = confusion_matrix(Y_test, Y_predict)
#print(c)
#sns.heatmap(c, annot=True, fmt=".1f")
#plt.xlabel('Actual');
#plt.ylabel('Predicted');


# In[54]:


#from sklearn.metrics import RocCurveDisplay
#RocCurveDisplay.from_predictions(Y_test, Y_predict, pos_label = 3)
#plt.show()


# # SKlearn NN

# In[55]:


get_ipython().run_cell_magic('time', '', "from sklearn.neural_network import MLPClassifier\n\nmodel = MLPClassifier(solver='adam',random_state=1, hidden_layer_sizes=(30,60), activation='relu')\nmodel.fit(X_train,Y_train)")


# In[56]:


Y_predicted = model.predict(X_test)


# In[57]:


print(classification_report(Y_test, Y_predicted))


# In[58]:


accuracy_score(Y_test,Y_predicted)


# In[59]:


from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(Y_test, Y_predicted, pos_label = 3)
plt.show()


# # Neural network

# In[60]:


# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
import seaborn as sns
# Keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
import keras.backend as K
from keras.metrics import RootMeanSquaredError 
# Train-Test
from sklearn.model_selection import train_test_split
# Scaling data
from sklearn.preprocessing import StandardScaler
# Classification Report
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers


# In[61]:


X_train.shape


# In[62]:


Y_train.shape


# In[63]:


get_ipython().run_cell_magic('time', '', 'input_shape = (6,)\nmodeltf = keras.Sequential(\n    [\n        Dense(60,input_shape = input_shape, activation="relu", name="layer1"),\n        Dense(30, activation="relu", name="layer2"),\n        Dense(60, activation="sigmoid", name="layer3"),\n        Dense(60, activation="softmax", name="layer4")\n    ])\n    ')


# In[64]:


modeltf.summary()


# In[65]:


get_ipython().run_cell_magic('time', '', 'modeltf.compile(Adam(learning_rate=0.01), "sparse_categorical_crossentropy", metrics = ["accuracy", "RootMeanSquaredError"])')


# In[66]:


get_ipython().run_cell_magic('time', '', 'history = modeltf.fit(X_train, Y_train, verbose=2, epochs=200, batch_size = 32, validation_data = (X_test,Y_test))')


# In[67]:


# Plotting Results
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.title('Training and validation accuracy')
plt.legend()
fig = plt.figure()
fig.savefig('acc.png')


plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss')

plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# # Gaussian Naive Bayes

# In[68]:


from sklearn.naive_bayes import GaussianNB


# In[69]:



gnb = GaussianNB()


# In[70]:


get_ipython().run_cell_magic('time', '', 'Y_pred = gnb.fit(X_train, Y_train).predict(X_test)')


# In[71]:


accuracy_score(Y_test,Y_pred)


# In[72]:


print(classification_report(Y_test, Y_pred))


# In[73]:


from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(Y_test, Y_pred, pos_label = 3)
plt.show()


# In[74]:


c = confusion_matrix(Y_test, Y_pred)
print(c)
sns.heatmap(c, annot=True, fmt=".1f")
plt.xlabel('Actual');
plt.ylabel('Predicted');


# # PCA

# In[75]:


lf3.drop(['Failure_type'], axis = 1)


# In[76]:


from sklearn.decomposition import PCA


# In[77]:


x = np.array([lf3.drop(['Failure_type'], axis = 1)])


# In[78]:


pca = PCA(n_components=3)


# In[79]:


d=pca.fit_transform(X)


# In[80]:


print(pca.explained_variance_ratio_)

pca_df = pd.DataFrame(data=d,columns = ['PC1', 'PC2', 'PC3'])


# In[81]:


PCAplot = np.arange(pca.n_components) + 1
plt.plot(PCAplot, pca.explained_variance_ratio_, 'ro-', linewidth = 2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance_ratio')
plt.show()


# In[82]:


pca_df


# In[83]:


pca_df["Failure_type"] = lf5


# In[84]:


pca_df


# In[85]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
pca_df.Failure_type = le.fit_transform(pca_df.Failure_type)
pca_df.head()


# In[86]:


X = pca_df.drop(['Failure_type'], axis = 1)
Y = pca_df['Failure_type']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size =0.33,  random_state = 2)


# # Decision Tree

# In[87]:


get_ipython().run_cell_magic('time', '', "dtc = DecisionTreeClassifier(criterion='gini', max_depth = 5, splitter='best')\ndtc = dtc.fit(X_train, Y_train)")


# In[88]:


predictions = dtc.predict(X_test)


# In[89]:


accuracy_score(Y_test,predictions)


# In[90]:


c = confusion_matrix(Y_test, predictions)


# In[91]:


c = confusion_matrix(Y_test, predictions)
print(c)
sns.heatmap(c, annot=True, fmt=".1f")
plt.xlabel('Actual');
plt.ylabel('Predicted');


# In[92]:


print(classification_report(Y_test, predictions))


# In[93]:


from sklearn import tree
plt.figure(figsize = (12,12))
tree.plot_tree(dtc,filled = True, class_names = True);


# In[ ]:





# # SVM

# In[94]:


#%%time
#from sklearn.svm import SVC

#model = SVC(kernel="poly", degree=8, gamma = 'auto')

#model.fit(X_train,Y_train)
#Y_predict = model.predict(X_test)


# In[95]:


#accuracy_score(Y_test,Y_predict)


# In[96]:


#print(classification_report(Y_test, Y_predict))


# In[97]:



#c = confusion_matrix(Y_test, Y_predict)
#print(c)
#sns.heatmap(c, annot=True, fmt=".1f")
#plt.xlabel('Actual');
#plt.ylabel('Predicted');


# # Random forest

# In[98]:


get_ipython().run_cell_magic('time', '', 'rfc = RandomForestClassifier(n_estimators=500, max_depth=4, random_state = 2)\nrfc.fit(X_train, Y_train)')


# In[99]:


Y_predicts = rfc.predict(X_test)


# In[100]:


accuracy_score(Y_test,Y_predicts)


# In[101]:


print(classification_report(Y_test, Y_predicts))


# # Neural network

# In[102]:


X_train.shape


# In[103]:


Y_train.shape


# In[104]:


get_ipython().run_cell_magic('time', '', 'input_shape = (3,)\nmodel = keras.Sequential(\n    [\n        Dense(60,input_shape = input_shape, activation="relu", name="layer1"),\n        Dense(30, activation="relu", name="layer2"),\n        Dense(60, activation="sigmoid", name="layer3"),\n        Dense(60, activation="softmax", name="layer4")\n    ])\n    ')


# In[105]:


model.summary()


# In[106]:


model.compile(Adam(learning_rate=0.01), "sparse_categorical_crossentropy", metrics = ["accuracy"])


# In[107]:


history = model.fit(X_train, Y_train, verbose=2, epochs=200, batch_size = 32, validation_data = (X_test,Y_test))


# In[108]:


# Plotting Results
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.title('Training and validation accuracy')
plt.legend()
fig = plt.figure()
fig.savefig('accPCA.png')


plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss')

plt.legend()
plt.show()


# # Gausian Bayes

# In[109]:


gnb = GaussianNB()


# In[110]:


get_ipython().run_cell_magic('time', '', 'Y_pred = gnb.fit(X_train, Y_train).predict(X_test)')


# In[111]:


accuracy_score(Y_test,Y_pred)


# # KNN

# In[112]:


neigh = KNeighborsClassifier(n_neighbors=3,weights='distance',leaf_size=30)
neigh.fit(X_train, Y_train)


# In[113]:


print(classification_report(Y_test, Y_pred))


# In[114]:


accuracy_score(Y_test,Y_pred)


# In[ ]:




