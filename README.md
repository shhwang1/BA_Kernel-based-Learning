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
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/198869856-10cec67e-30d9-40bf-8e28-c0b98b13f123.png"></p>
