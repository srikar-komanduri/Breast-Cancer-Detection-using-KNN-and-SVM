import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd


# loading the dataset
url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names=['id','CT','USC','UCSh','MA','SES','BN','BC','NN','M','C']
df=pd.read_csv(url,names=names)

#replaces '?' with -99999. i.e ignores this value
df.replace('?',-99999,inplace=True)

#drops the id column
df.drop(['id'],1,inplace=True)

# Visualizing the dataset
'''
print(df.axes)
print(df.shape)
print(df.loc[101])


print(df.hist(figsize=(10,10)))
plt.show()


scatter_matrix(df,figsize=(10,10))
plt.show()'''

#dividing the dataset into X and Y for training set


X = np.array(df.drop(['C'],1))
Y = np.array(df['C'])

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#specifying testing options

seed=8
scoring='accuracy'

#defining models to train

models=[]
models.append(('KNN',KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM',SVC()))

#evaluate each model in turn 
names=[]
results=[]

for name , model in models:
  kfold=model_selection.KFold(n_splits=10)
  cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
  results.append(cv_results)
  names.append(name)

  print("%s: %f (%f) " % (name, cv_results.mean(), cv_results.std()) )

#make predictions

for name,model in models:
  model.fit(X_train,Y_train)
  predictions=model.predict(X_test)
  print(name)
  print(accuracy_score(Y_test,predictions))
  print(classification_report(Y_test,predictions))
  




