import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
        
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics 
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('D:/ML/Python2023/PhishingUrlDetection/phishing.csv')
data.head()
data.columns
len(data.columns)
data.isnull().sum()
X = data.drop(["class","Index"],axis =1)
y = data["class"]
fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(data.corr(), annot=True,cmap='viridis')
plt.title('Correlation between different features', fontsize = 15, c='black')
plt.show()
corr=data.corr()
corr.head()
corr['class']=abs(corr['class'])
corr.head()
incCorr=corr.sort_values(by='class',ascending=False)
incCorr.head()
incCorr['class']
tenfeatures=incCorr[1:11].index
twenfeatures=incCorr[1:21].index
ML_Model = []
accuracy = []
f1_score = []
precision = []

def storeResults(model, a,b,c):
  ML_Model.append(model)
  accuracy.append(round(a, 3))
  f1_score.append(round(b, 3))
  precision.append(round(c, 3))
def KNN(X):
  x=[a for a in range(1,10,2)]
  knntrain=[]
  knntest=[]
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
  X_train.shape, y_train.shape, X_test.shape, y_test.shape
  for i in range(1,10,2):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_train_knn = knn.predict(X_train)
    y_test_knn = knn.predict(X_test)
    acc_train_knn = metrics.accuracy_score(y_train,y_train_knn)
    acc_test_knn = metrics.accuracy_score(y_test,y_test_knn)
    print("K-Nearest Neighbors with k={}: Accuracy on training Data: {:.3f}".format(i,acc_train_knn))
    print("K-Nearest Neighbors with k={}: Accuracy on test Data: {:.3f}".format(i,acc_test_knn))
    knntrain.append(acc_train_knn)
    knntest.append(acc_test_knn)
    print()
  import matplotlib.pyplot as plt
  plt.plot(x,knntrain,label="Train accuracy")
  plt.plot(x,knntest,label="Test accuracy")
  plt.legend()
  plt.show()
Xmain=X
Xten=X[tenfeatures]
Xtwen=X[twenfeatures]
KNN(Xmain)
KNN(Xten)
KNN(Xtwen)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

y_train_knn = knn.predict(X_train)
y_test_knn = knn.predict(X_test)

acc_train_knn = metrics.accuracy_score(y_train,y_train_knn)
acc_test_knn = metrics.accuracy_score(y_test,y_test_knn)

f1_score_train_knn = metrics.f1_score(y_train,y_train_knn)
f1_score_test_knn = metrics.f1_score(y_test,y_test_knn)

precision_score_train_knn = metrics.precision_score(y_train,y_train_knn)
precision_score_test_knn = metrics.precision_score(y_test,y_test_knn)
storeResults('K-Nearest Neighbors',acc_test_knn,f1_score_test_knn,precision_score_train_knn)
def SVM(X, y):
    x=[a for a in range(1,10,2)]
    svmtrain=[]
    svmtest=[]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    X_train.shape, y_train.shape, X_test.shape, y_test.shape
    from sklearn.svm import SVC
    for i in range(1,10,2):
        svm = SVC(kernel='linear', C=i)
        svm.fit(X_train, y_train)
        y_train_svm = svm.predict(X_train)
        y_test_svm = svm.predict(X_test)
        acc_train_svm = metrics.accuracy_score(y_train, y_train_svm)
        acc_test_svm = metrics.accuracy_score(y_test, y_test_svm)
        print("SVM with C={}: Accuracy on training Data: {:.3f}".format(i,acc_train_svm))
        print("SVM with C={}: Accuracy on test Data: {:.3f}".format(i,acc_test_svm))
        svmtrain.append(acc_train_svm)
        svmtest.append(acc_test_svm)
        print()
    import matplotlib.pyplot as plt
    plt.plot(x,svmtrain,label="Train accuracy")
    plt.plot(x,svmtest,label="Test accuracy")
    plt.legend()
    plt.show()
Xmain=X
Xten=X[tenfeatures]
Xtwen=X[twenfeatures]

SVM(Xmain,y)
SVM(Xtwen,y)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train, y_train)


y_train_svm = svm.predict(X_train)
y_test_svm = svm.predict(X_test)


acc_train_svm = metrics.accuracy_score(y_train, y_train_svm)
acc_test_svm = metrics.accuracy_score(y_test, y_test_svm)

f1_score_train_svm = metrics.f1_score(y_train, y_train_svm)
f1_score_test_svm = metrics.f1_score(y_test, y_test_svm)

precision_score_train_svm = metrics.precision_score(y_train, y_train_svm)
precision_score_test_svm = metrics.precision_score(y_test, y_test_svm)

print("SVM with C={}: Accuracy on training data: {:.3f}".format(1, acc_train_svm))
print("SVM with C={}: Accuracy on test data: {:.3f}".format(1, acc_test_svm))
print("SVM with C={}: F1 score on training data: {:.3f}".format(1, f1_score_train_svm))
print("SVM with C={}: F1 score on test data: {:.3f}".format(1, f1_score_test_svm))
print("SVM with C={}: Precision on training data: {:.3f}".format(1, precision_score_train_svm))
print("SVM with C={}: Precision on test data: {:.3f}".format(1, precision_score_test_svm))
storeResults('Support Vector Machines',acc_test_svm,f1_score_test_svm,precision_score_train_svm)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)
gbc.fit(X_train,y_train)
y_train_gbc = gbc.predict(X_train)
y_test_gbc = gbc.predict(X_test)
acc_train_gbc = metrics.accuracy_score(y_train,y_train_gbc)
acc_test_gbc = metrics.accuracy_score(y_test,y_test_gbc)
print("Gradient Boosting Classifier : Accuracy on training Data: {:.3f}".format(acc_train_gbc))
print("Gradient Boosting Classifier : Accuracy on test Data: {:.3f}".format(acc_test_gbc))
print()

f1_score_train_gbc = metrics.f1_score(y_train,y_train_gbc)
f1_score_test_gbc = metrics.f1_score(y_test,y_test_gbc)

precision_score_train_gbc = metrics.precision_score(y_train,y_train_gbc)
precision_score_test_gbc = metrics.precision_score(y_test,y_test_gbc)

storeResults('Gradient Boosting Classifier',acc_test_gbc,f1_score_test_gbc,precision_score_train_gbc)
df = pd.DataFrame({
    'Modelname': ML_Model,
    'Accuracy Score': accuracy,
    'F1 Score': f1_score,
    'Precision Score': precision
})
df.set_index('Modelname', inplace=True)

# plot the scores for each model

fig, ax = plt.subplots(figsize=(10,10))
df.plot(kind='bar', ax=ax)
ax.set_xticklabels(df.index, rotation=0)
ax.set_ylim([0.9, 1])
ax.set_yticks([0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1])
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Scores')
plt.show()
