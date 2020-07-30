#!/usr/bin/env python
# coding: utf-8

# # Description
# ### The objective of the dataset is to analyze the quality of red and white variants of the Portuguese "Vinho Verde" wine. The datasets consists of certain input variables(based on physicochemical tests) and one target variable, quality -score between 0 and 10(based on sensory data).

# ### Importing libraries and dataset

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[4]:


df=pd.read_csv(r'C:\Users\Amina\OneDrive\Desktop\projects\ML projects\winequality-red.csv')


# In[5]:


df.head()


# ### Descriptive statistics

# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.shape


# In[10]:


df['quality'].value_counts()


# ### Visualization

# In[11]:


import seaborn as sns
sns.countplot(x = 'quality',data = df)


# In[12]:


f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax,annot=True)


# The countplot tellus that there is a high correlation between quality and alcohol, sulphate content,citric acid and fixed acidity. We can select these features to accept input from the user and predict the outcome.
# 

# ### Data pre-processing

# In[13]:


new_df=df.replace(0,np.NaN)


# In[14]:


new_df.isnull().sum()


# In[15]:


new_df["citric acid"].fillna(new_df["citric acid"].mean(), inplace = True)


# In[16]:


new_df.describe().T


# In[17]:


# converting the response variables(3-7) as binary response variables that is either good or bad

names = ['bad', 'good']
bins = (2, 6.5, 8)

df['quality'] = pd.cut(df['quality'], bins = bins, labels = names)


# In[18]:


df['quality'].value_counts()


# In[21]:


#We have now labelled the quality into good and bad,now to convert them into numerical values

from sklearn.preprocessing import LabelEncoder
label_quality=LabelEncoder()
df['quality']= label_quality.fit_transform(df['quality'])
df['quality'].value_counts()


# In[22]:


sns.countplot(df['quality'])


# In[ ]:





# In[ ]:





# In[ ]:





# ### Feature engineering
# 

# In[89]:


#FeatureSelection
X=df.iloc[:,:11].values
y=df.iloc[:,11].values
#splitting X and y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size = 0.20, random_state = 44 )


# In[90]:


#Checking dimensions
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[91]:


# standard scaling 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ### Model Building

# In[101]:


# Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 44)
logreg.fit(X_train, y_train)


# In[102]:


# Support Vector Classifier Algorithm
from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 44)
svc.fit(X_train, y_train)


# In[103]:


# Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)


# In[104]:


# Decision tree Algorithm
from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 44)
dectree.fit(X_train, y_train)


# In[105]:


# Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 44)
ranfor.fit(X_train, y_train)


# In[107]:


# Making predictions on test dataset
Y_pred_logreg = logreg.predict(X_test)
#Y_pred_knn = knn.predict(X_test)
Y_pred_svc = svc.predict(X_test)
Y_pred_nb = nb.predict(X_test)
Y_pred_dectree = dectree.predict(X_test)
Y_pred_ranfor = ranfor.predict(X_test)


# In[ ]:





# ### Model Evaluation

# In[109]:


# Evaluating using accuracy_score metric
from sklearn.metrics import accuracy_score
accuracy_logreg = accuracy_score(y_test, Y_pred_logreg)
#accuracy_knn = accuracy_score(y_test, Y_pred_knn)
accuracy_svc = accuracy_score(y_test, Y_pred_svc)
accuracy_nb = accuracy_score(y_test, Y_pred_nb)
accuracy_dectree = accuracy_score(y_test, Y_pred_dectree)
accuracy_ranfor = accuracy_score(y_test, Y_pred_ranfor)


# In[110]:


# Accuracy on test set
print("Logistic Regression: " + str(accuracy_logreg * 100))
#print("K Nearest neighbors: " + str(accuracy_knn * 100))
print("Support Vector Classifier: " + str(accuracy_svc * 100))
print("Naive Bayes: " + str(accuracy_nb * 100))
print("Decision tree: " + str(accuracy_dectree * 100))
print("Random Forest: " + str(accuracy_ranfor * 100))


# Random forest gives the best accuracy score

# In[111]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred_ranfor)
cm


# In[112]:


# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)


# In[113]:


# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, Y_pred_ranfor))


# In[ ]:




