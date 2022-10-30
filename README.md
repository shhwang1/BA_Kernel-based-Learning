# BA_Kernel-based-Learning Tutorial
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/198869320-63f20533-de48-4ea3-9adb-1fa7e2298bd3.png" width="600" height="300"></p>

## Purpose of Tutorial
There are data with binary classes and data with various classes in the world. Finding a classifier that perfectly distinguishes classes for these data is a key task in classification problems. However, it is a very difficult problem to find linear classifiers just for data with binary classes, not to mention multi-class data. This tutorial contains the contents of kernel-based learning for the purpose of discovering the classifier. In this tutorial, support vector machine and support vector regression are described as the methodology of kernel-base learning. In addition, various hyperparameters are adjusted for various datasets, and analysis of experimental results is the main content.   
   
The table of contents is as follows.

___
### Support Vector Machine(SVM)

#### 1. Definition
   
#### 2. Cases of SVM 
   
#### 3. Python Code  
   
#### 4. Analysis   
___
### Support Vector Regression(SVR)

#### 1. Definition
   
#### 2. Python Code  
   
#### 3. Analysis 
___

## Dataset
We use 4 datasets (abalone, Diabetes, PersonalLoan, WineQuality)

abalone dataset : <https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset>     
Diabetes dataset : <https://www.kaggle.com/datasets/mathchi/diabetes-data-set>   
PersonalLoan dataset : <https://www.kaggle.com/datasets/teertha/personal-loan-modeling>   
WineQuality datset : <https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009>    

Each dataset is a dataset for classification with a specific class value as y-data.   
   
In all methods, data is used in the following form.
``` C
import argparse

def Parser1():
    parser = argparse.ArgumentParser(description='1_Dimensionality Reduction')

    # data type
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--data-type', type=str, default='abalone.csv',
                        choices = ['abalone.csv', 'BankNote.csv', 'PersonalLoan.csv', 'WineQuality.csv', 'Diabetes.csv'])
                        
data = pd.read_csv(args.data_path + args.data_type)

X_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]
```

___

## Support Vector Machine (SVM)   
   
### 1. Definition
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/198869856-10cec67e-30d9-40bf-8e28-c0b98b13f123.png"  width="500" height="300"></p>   
   
In general, Support Vector Machine refers to a methodology for separating data using a line called a 'Hyperplane'.   
However, in the case of the above picture, it is possible to create a lot of hyperplanes that separate the data well just by properly changing the slope of the line.    
Accordingly, there may be a question of which hyperplane is the optimal hyperplane.    
As shown in the picture below, SVM uses the hyperplane that maximizes 'Margin' as the optimal hyperplane.   
   
   
   <p align="center"><img src="https://user-images.githubusercontent.com/115224653/198870201-f2d22a11-37a8-4030-8d7f-6b9f3ba7b60b.png"  width="500" height="300"></p>   
   
However, data that is completely linearly separated like the data above, as you may have noticed, rarely exists in reality.   
Therefore, in general, some errors are allowed, non-linear classifiers are used, and non-linear classifiers and errors are used simultaneously.   
   
Arrange each case in the table of contents below.   

___   
### 2. Cases of SVM   
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/198870628-2a6bea54-48c4-412f-8e43-7dbeb6190a08.png"  width="1000" height="300"></p> 

The first case, Linearly Hard Margin, is a very basic form of SVM. This method assumes that the data is linearly separated and has a feature that does not allow any errors. As mentioned above, it can be said that it is a very difficult method to apply in reality.   

The second case, Linearly Soft Margin, is a method of maintaining a linear classifier shape but allowing a certain amount of error. How much error is allowed can be set through the "C" value, which is a hyperparameter, and the performance difference according to the "C" value will be verified in the 'Analysis' table of contents later.

Thr fourth case(we do not care third case), Non-linearly Soft Margin, The fourth method is a non-linear method that applies a soft margin method that allows errors and uses a non-linear mapping technique to map on a high-dimensional space. We compare the performance according to the 'rbf' method and the 'polynomial' method, which are widely known non-linear methods in later experiments.   
   
   <p align="center"><img src="https://user-images.githubusercontent.com/115224653/198880148-4e56d853-ca9f-4ca9-84c9-7e14a6983dc8.png"  width="900" height="400"></p>   

___
### 3. Python Code
``` C
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Linearly_Hard_Case(args):
    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = args.split_size, shuffle=True, random_state = args.seed)

    clf = svm.LinearSVC(C=1e-10, max_iter = args.max_iter, random_state = args.seed) # 1e-10 means very small value which is close to "0" (There's no Hard-margin method in sklearn)
    clf.fit(X_train, y_train)
    print('Training SVM : Linearly-Hard Margin SVM.....')

    # Prediction
    pred_clf = clf.predict(X_test)
    accuracy_clf = accuracy_score(y_test, pred_clf)
    print('Linearly-Hard margin SVM accuracy : ', accuracy_clf)
```    

We used the scikit-learn package to implement SVM. The first case, Linearly-Hard case, does not have a suitable hyperparameter to deal with in Analysis as mentioned above. Meanwhile, an argument called 'max_iter' was found in the LinearSVC Method, which contained information indicating how many times the SVM would be trained repeatedly. So we decided to compare the experimental performance according to 'max_iter' among the few parameters.
