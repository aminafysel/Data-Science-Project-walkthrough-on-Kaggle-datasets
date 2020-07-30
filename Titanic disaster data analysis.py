#!/usr/bin/env python
# coding: utf-8

# # A detailed statistical analysis of Titanic data set along with Machine learning model implementation

# #### The sinking of the Titanic is one of the most infamous shipwrecks in history.
# 
# #### On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# #### While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. Here I am doing an attempt to explore the same.

# ## Loading Data

# In[1]:


# data processing
import pandas as pd

## linear algebra
import numpy as np

# data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
 
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.metrics import confusion_matrix #for confusion matrix


# In[2]:


df=pd.read_csv(r'C:\Users\Amina\OneDrive\Desktop\projects\ML projects\titanic.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# ## Data Cleaning

# In[7]:


missing=df.isnull().sum().sort_values(ascending=False)
missing.head()


# In[8]:


percent = round(df.isnull().sum().sort_values(ascending = False) * 100 /len(df),2)


# In[9]:


missing_percentage=pd.concat([missing, percent], axis=1, keys=['Missing','Percent'])


# In[10]:


missing_percentage.head(15)


# In[11]:


drop_column = ['Body','Cabin',]
df.drop(drop_column, axis= 1, inplace = True)


# In[12]:


df.head(5)


# In[13]:


drop_column = ['Lifeboat']
df.drop(drop_column, axis= 1, inplace = True)


# In[14]:


df['Age'].fillna(df['Age'].median(), inplace = True)
df['Survived'].fillna(df['Survived'].mode()[0], inplace = True)


# In[15]:


df.isnull().sum()


# In[16]:


df1 = df.dropna()
print(df1)


# In[17]:


df1.isnull().sum()


# ## Data Visualization

# In[18]:


sns.set(style="darkgrid")
plt.figure(figsize = (8, 5))
graph= sns.countplot(x='Survived', hue="Survived", data=df1)


# ### Male and female survival count

# In[19]:


plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Sex", hue ="Survived", data = df1)


# ### Embarked and p-class vs survival
# 
# #### Embarked: From which location passenger go on board to Titanic.
# 
# C = Cherbourg
# Q = Queenstown
# S = Southampton
# 
# 
# #### PClass: Passenger belongs to which class.
# 
# 1st = Upper
# 2nd = Middle
# 3rd = Lower

# In[20]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4)) 
x = sns.countplot(df1['Pclass'], ax=ax[0])
y = sns.countplot(df1['Embarked'], ax=ax[1])

fig.show()


# #### Since it does not matter from where didi the passenger board ,as it is more important that the passenger was currently on Titanic,as we know that At 2:20 a.m. on April 15, 1912, the British ocean liner Titanic sinks into the North Atlantic Ocean.We can use Embarked as feature here for getting high accuracy but logically its doesn't matter. so we drop it out.

# In[21]:


drop_column = ['Embarked']
df1.drop(drop_column, axis=1, inplace = True)


# In[22]:


df1.head()


# ### Passenger class impact on survival

# In[23]:


plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Pclass", hue ="Survived", data = df1)


# ##### Here we see clearly, that Pclass is contributing to a persons chance of survival, especially if this person was in class 1, and class 3 proved the least chances of survival. We will create another pclass plot below

# In[24]:


plt.figure(figsize = (8, 5))
sns.barplot(x='Pclass', y='Survived', data=df1)


# ### Sibsp and parch vs survived

# In[25]:


axes = sns.factorplot('SibSp','Survived', 
                      data=df1, aspect = 2.5, )


# In[26]:


axes = sns.factorplot('Parch','Survived', 
                      data=df1, aspect = 2.5, )


# ##### Both the above plots describe that the chances of survival goes down as the member in a family increases

# ### Age vs survived

# In[27]:


df1['Age_bin'] = pd.cut(df1['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
    
plt.figure(figsize = (8, 5))
sns.barplot(x='Age_bin', y='Survived', data=df1)


# In[28]:


plt.figure(figsize = (8, 5))
sns.countplot(x='Age_bin', hue='Survived', data=df1)


# ##### Children below 12 ears of age had higher chances of survival,as we can assume that parents and siblings might have saved the younger ones before themselves.

# ### Fare vs survived

# In[29]:


df1['Fare_bin'] = pd.cut(df1['Fare'], bins=[0,10,50,100,550], labels=['Low_fare','median_fare','Average_fare','high_fare'])
plt.figure(figsize = (8, 5))
sns.countplot(x='Pclass', hue='Fare_bin', data=df1)


# In[30]:


sns.barplot(x='Fare_bin', y='Survived', data=df1)


# ##### people in Pclass 1 with high fare had a higher survival chance, and people with low to average fare had a very low survival rate

# ### Correlation matrix

# In[31]:


df1.head()


# In[32]:


pd.DataFrame(abs(df1.corr()['Survived']).sort_values(ascending = False))


# In[33]:


f, ax = plt.subplots(figsize=(10, 8))
corr = df1.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax,annot=True)


# ### Feature engineering

# In[34]:


df1.info()


# In[36]:


# Convert ‘Sex’ feature into numeric.
genders = {"male": 0, "female": 1}

df1['Sex'] = df1['Sex'].map(genders)
df1['Sex'].value_counts()


# In[37]:


drop_column = ['Age_bin','Fare','Name','Ticket', 'PassengerId','WikiId','Name_wiki','Age_wiki','Hometown','Boarded','Destination','Fare_bin']
df1.drop(drop_column, axis=1, inplace = True)


# In[38]:


df1.head()


# ### Predictive modelling

# In[39]:


all_features = df1.drop("Survived",axis=1)
Targete = df1["Survived"]
X_train,X_test,y_train,y_test = train_test_split(all_features,Targete,test_size=0.3,random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[40]:


# standard scaling 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[58]:


# Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 22)
logreg.fit(X_train, y_train)


# In[50]:


# Support Vector Classifier Algorithm
from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 22)
svc.fit(X_train, y_train)


# In[43]:


# Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)


# In[51]:


# Decision tree Algorithm
from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 22)
dectree.fit(X_train, y_train)


# In[52]:


# Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 22)
ranfor.fit(X_train, y_train)


# In[53]:


# Making predictions on test dataset
Y_pred_logreg = logreg.predict(X_test)
#Y_pred_knn = knn.predict(X_test)
Y_pred_svc = svc.predict(X_test)
Y_pred_nb = nb.predict(X_test)
Y_pred_dectree = dectree.predict(X_test)
Y_pred_ranfor = ranfor.predict(X_test)


# In[54]:


# Evaluating using accuracy_score metric
from sklearn.metrics import accuracy_score
accuracy_logreg = accuracy_score(y_test, Y_pred_logreg)
#accuracy_knn = accuracy_score(y_test, Y_pred_knn)
accuracy_svc = accuracy_score(y_test, Y_pred_svc)
accuracy_nb = accuracy_score(y_test, Y_pred_nb)
accuracy_dectree = accuracy_score(y_test, Y_pred_dectree)
accuracy_ranfor = accuracy_score(y_test, Y_pred_ranfor)


# In[55]:


# Accuracy on test set
print("Logistic Regression: " + str(accuracy_logreg * 100))
#print("K Nearest neighbors: " + str(accuracy_knn * 100))
print("Support Vector Classifier: " + str(accuracy_svc * 100))
print("Naive Bayes: " + str(accuracy_nb * 100))
print("Decision tree: " + str(accuracy_dectree * 100))
print("Random Forest: " + str(accuracy_ranfor * 100))


# ##### Logistic regression gives the best accuracy score

# ### Evaluation

# In[59]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred_logreg)
cm


# In[60]:


# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)


# In[61]:


# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, Y_pred_ranfor))


# In[ ]:




