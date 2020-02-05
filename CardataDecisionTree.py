# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:35:38 2020

@author: new
"""

# Classify the cars good or very good
#using the decision tree classifier
 
# import dependencies/ libraries

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# import the data
df = pd.read_csv('C:/Users/new/Desktop/cardata.csv')
df.shape

# renaming the columns as per dataset

df =df.rename(columns={"Column1": "buying", "Column2": "maint","Column3":"doors","Column4":"persons","Column5":"lug_boot","Column6":"safety","Column7":"values"})

print(df.head())

# import label encoder to convert into categorical variables

from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
  
df['buying']= le.fit_transform(df['buying'])
df['maint']= le.fit_transform(df['maint'])
df['lug_boot']= le.fit_transform(df['lug_boot'])
df['safety']= le.fit_transform(df['safety'])
df['doors']= le.fit_transform(df['doors'])
df['persons']= le.fit_transform(df['persons'])
# splitting data into independent variable and dependent variables

x_train = df.loc[:,'buying':'safety'] # importing all rows and columns only from buying to safety
y_train = df.loc[:,'values'] # get all rows in dataset and only values column

# create the decision tree classifier

tree = DecisionTreeClassifier(max_leaf_nodes=5,random_state=0)

# train the model

tree.fit(x_train,y_train)

# make our prediction
# imput : buying= very hign, maint=high, doors=2,persons=2, lug_boot=med, safety=high(3) putting categorical values associated with each 

prediction = tree.predict([[4,3,2,2,2,3]])

# print the prediction
print(prediction)

