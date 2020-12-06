#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


data = pd.read_csv('data.csv')
data.head()


# In[3]:


data.shape        #Count the number of rows and columns in the data file


# In[4]:


data['diagnosis']


# In[5]:


data.isna().sum()    #Count the empty in each column


# In[6]:


data = data.dropna(axis=1)    #This drops the column Unnamed // drop the column with all missing values


# In[7]:


data.shape                  #new count of the number of rows and cols after dropping


# In[8]:


count=data['diagnosis'].value_counts()      #count of the number of 'M' & 'B' cells using value_counts()
count


# In[9]:


fig = plt.figure()
ax = fig.add_axes([0,0,0.7,1])
langs = ["B", "M"]
students = data['diagnosis'].value_counts()
ax.bar(langs,students)
plt.xlabel("daignosis") 
plt.ylabel("No. of count") 
plt.title("diagnosis graph") 
plt.show() 


# In[10]:


sns.countplot(data['diagnosis'],label="Count")            #Visualize this count   different method


# In[11]:


data.dtypes        #to see the data types of columns


# In[12]:


from sklearn.preprocessing import LabelEncoder                  #labeling categorical data values
labelencoder_Y = LabelEncoder()
data.iloc[:,1]= labelencoder_Y.fit_transform(data.iloc[:,1].values)                #0-M   1-B
print(labelencoder_Y.fit_transform(data.iloc[:,1].values))


# In[13]:


sns.pairplot(data, hue="diagnosis")


# In[14]:


plt.figure(figsize = (20,8))
sns.countplot(data['radius_mean'])


# In[15]:


data.head()


# In[16]:


data.corr()        #finding correlation


# In[17]:


plt.figure(figsize=(20,20))  
sns.heatmap(data.corr(), annot=True, fmt='.0%')


# In[18]:


## pair of independent variables with correlation greater than 0.5
k = data.corr()
z = [[str(i),str(j)] for i in k.columns for j in k.columns if (k.loc[i,j] >abs(0.5))&(i!=j)]
z, len(z)


# In[19]:


X = data.iloc[:, 2:31].values 
Y = data.iloc[:, 1].values 


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 101)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# # Scaling the data

# In[21]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize = True)
lr.fit(X_train, Y_train)


# In[22]:


lr.coef_


# In[23]:


predictions = lr.predict(X_test)


# In[24]:


lr.score(X_test, Y_test)


# # feature Scaling using sklearn

# In[25]:


data.describe()


# In[26]:


from sklearn import preprocessing
scale = preprocessing.StandardScaler()


# In[27]:


X_train = scale.fit_transform(X_train)
X_train


# In[28]:


X_test = scale.fit_transform(X_test)
X_test


# Simple Predictive model

# In[29]:


data['diagnosis_mean'] =  data['diagnosis'].mean()
data['diagnosis_mean'].head()


# In[30]:


plt.figure(dpi = 70)
k = range(0, len(data))
plt.scatter(k, data['diagnosis'].sort_values(),color='red', label='Actual diagnosis')
plt.plot(k, data['diagnosis_mean'].sort_values(), color = 'green', label = 'mean diagnosis')
plt.xlabel('points')
plt.ylabel('diagnosis')
plt.legend()


# In[31]:


data.keys()


# In[32]:


# visualize correlation barplot
plt.figure(figsize = (16,5))
ax = sns.barplot(data.corrwith(data.diagnosis).index, data.corrwith(data.diagnosis))
ax.tick_params(labelrotation = 90)


# # MODEL

# In[33]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ## Logistic Regression 

# In[34]:


def models(X_train,Y_train):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, Y_train)
    #print model accuracy on the training data.
    print('Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    return log

model = models(X_train,Y_train)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, model.predict(X_test))
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
print(cm)
print('Testing Accuracy = "{}!"'.format((TP + TN) / (TP + TN + FN + FP)))
print()
print()

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print( classification_report(Y_test, model.predict(X_test)) )
#Another way to get the models accuracy on the test data
print( accuracy_score(Y_test, model.predict(X_test)))
print()#Print a new line


# In[35]:


#print prediction of logistic regression
print("Prediction of Logistic Regression")
pred = model.predict(X_test)
print(pred)

#Print a space
print()

#Print the actual values
print("Normal Prediction")
print(Y_test)


# In[36]:


from sklearn import linear_model 
logreg = linear_model.LogisticRegression(random_state = 0) 
print("test accuracy: {} ".format(logreg.fit(X_train, Y_train).score(X_test, Y_test))) 
print("train accuracy: {} ".format(logreg.fit(X_train, Y_train).score(X_train, Y_train))) 


# In[37]:


## Pickle   0.986013986013986 
import pickle
 
# save model
pickle.dump(model,open('breast_cancer_detector.pickle', 'wb'))
 
# load model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))
 
# predict the output
y_pred = breast_cancer_detector_model.predict(X_test)
 
# confusion matrix
print('Confusion matrix of Logistic Regression model: \n',confusion_matrix(Y_test, y_pred),'\n')
 
# show the accuracy
print('Accuracy of Logistic Regression model = ',accuracy_score(Y_test, y_pred))


# # #Using KNeighborsClassifier

# In[38]:


def models(X_train,Y_train):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    knn.fit(X_train, Y_train)
    print('K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    return knn

model = models(X_train,Y_train)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, model.predict(X_test))
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
print(cm)
print('Testing Accuracy = "{}!"'.format((TP + TN) / (TP + TN + FN + FP)))
print()
print()


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print( classification_report(Y_test, model.predict(X_test)) )
#Another way to get the models accuracy on the test data
print( accuracy_score(Y_test, model.predict(X_test)))
print()#Print a new line


# In[39]:


#print prediction of using KNeighborsClassifier
print("Prediction of using KNeighborsClassifier")
pred = model.predict(X_test)
print(pred)

#Print a space
print()

#Print the actual values
print("Normal Prediction")
print(Y_test)


# # Using SVC linear

# In[40]:


def models(X_train,Y_train):
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X_train, Y_train)
    print('Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    return svc_lin

model = models(X_train,Y_train)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, model.predict(X_test))
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
print(cm)
print('Testing Accuracy = "{}!"'.format((TP + TN) / (TP + TN + FN + FP)))
print()
print()


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print( classification_report(Y_test, model.predict(X_test)) )
#Another way to get the models accuracy on the test data
print( accuracy_score(Y_test, model.predict(X_test)))
print()#Print a new line


# In[41]:


#print prediction of Using SVC linear
print("Prediction of Using SVC linear")
pred = model.predict(X_test)
print(pred)

#Print a space
print()

#Print the actual values
print("Normal Prediction")
print(Y_test)


# # Using SVC rbf

# In[42]:


def models(X_train,Y_train):
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train, Y_train)
    print('Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
    return svc_rbf

model = models(X_train,Y_train)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, model.predict(X_test))
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
print(cm)
print('Testing Accuracy = "{}!"'.format((TP + TN) / (TP + TN + FN + FP)))
print()
print()

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print( classification_report(Y_test, model.predict(X_test)) )
#Another way to get the models accuracy on the test data
print( accuracy_score(Y_test, model.predict(X_test)))
print()#Print a new line


# In[43]:


#print prediction of Using SVC rbf
print("Prediction of Using SVC rbf")
pred = model.predict(X_test)
print(pred)

#Print a space
print()

#Print the actual values
print("Normal Prediction")
print(Y_test)


# # Using GaussianNB 

# In[44]:


def models(X_train,Y_train):
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)
    print('Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
    return gauss

model = models(X_train,Y_train)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, model.predict(X_test))
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
print(cm)
print('Testing Accuracy = "{}!"'.format((TP + TN) / (TP + TN + FN + FP)))
print()
print()

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print( classification_report(Y_test, model.predict(X_test)) )
#Another way to get the models accuracy on the test data
print( accuracy_score(Y_test, model.predict(X_test)))
print()#Print a new line


# In[45]:


#print prediction of Using GaussianNB
print("Prediction of Using GaussianNB")
pred = model.predict(X_test)
print(pred)

#Print a space
print()

#Print the actual values
print("Normal Prediction")
print(Y_test)


# # Using DecisionTreeClassifier 

# In[46]:


def models(X_train,Y_train):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, Y_train)
    print('Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    return tree

model = models(X_train,Y_train)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, model.predict(X_test))
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
print(cm)
print('Testing Accuracy = "{}!"'.format((TP + TN) / (TP + TN + FN + FP)))
print()
print()


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print( classification_report(Y_test, model.predict(X_test)) )
#Another way to get the models accuracy on the test data
print( accuracy_score(Y_test, model.predict(X_test)))
print()#Print a new line


# In[47]:


#print prediction of Using DecisionTreeClassifier
print("Prediction of Using DecisionTreeClassifier")
pred = model.predict(X_test)
print(pred)

#Print a space
print()

#Print the actual values
print("Normal Prediction")
print(Y_test)


# # Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

# In[48]:


def models(X_train,Y_train):
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)
    print('Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
    return forest

model = models(X_train,Y_train)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, model.predict(X_test))
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
print(cm)
print('Testing Accuracy = "{}!"'.format((TP + TN) / (TP + TN + FN + FP)))
print()
print()

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print( classification_report(Y_test, model.predict(X_test)) )
#Another way to get the models accuracy on the test data
print( accuracy_score(Y_test, model.predict(X_test)))
print()#Print a new line


# In[49]:


#print prediction of Using RandomForestClassifier
print("Prediction of Using RandomForestClassifier")
pred = model.predict(X_test)
print(pred)

#Print a space
print()

#Print the actual values
print("Normal Prediction")
print(Y_test)


# In[50]:


#end 

