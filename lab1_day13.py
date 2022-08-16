import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier






df=pd.read_csv('C:/Users/bader/OneDrive/gitlesson/Day13-Lab1-KNN-LR/Social_Network_Ads.csv')
print(df.head())
#Define X by selecting only the age and EstimatedSalary, and y with purchased column
x=df[['Age','EstimatedSalary']]
print(x.head(),'\n')
y=df['Purchased']
print(y.head(),'\n')
#Print count of each label in Purchased column

print(y.value_counts(),'\n')
#Print Correlation of each feature in the dataset
Correlation =pd.DataFrame(data=df.corr())
print(Correlation,'\n')
#First: Logistic Regression model
#Split the dataset into Training set and Test set with test_size = 0.25 and random_state = 0
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.25,random_state=0)
model=LogisticRegression()
model.fit(x_tr,y_tr)
pred=model.predict(x_ts)
print('Print the prediction results: \n',pred)

#Create dataframe with the Actual Purchased and Predict Purchased
comp_result = pd.DataFrame(x)
comp_result['Actual_Purchased']=y_ts
comp_result=comp_result.dropna()
comp_result['Predict_Purchased']=pred
print('Create dataframe with the Actual Purchased and Predict Purchased\n',comp_result)




#Print Confusion Matrix and classification_report

print('LogisticRegression before Scaled\n',classification_report(y_ts,pred))

plot_confusion_matrix(model, X=x_ts, y_true=y_ts, cmap='Blues')
plt.show()

#Use StandardScaler() to improved performance and re-train your model
Scaler = StandardScaler()
x_tr = Scaler.fit_transform(x_tr)
x_ts=Scaler.transform(x_ts)
model.fit(x_tr,y_tr)
pred=model.predict(x_ts)
print('LogisticRegression after Scaled\n',classification_report(y_ts,pred))


#Try to Predicting a new result - e.g: person with Age = 30 and Salary = 90,000

print('#Try to Predicting a new result - e.g: person with Age = 30 and Salary = 90,000\n',model.predict(Scaler.transform([[30,90000]])),'\n')
print('#Try to Predicting a new result - e.g: person with Age = 40 and Salary = 90,000\n',model.predict(Scaler.transform([[40,90000]])),'\n')


#Second: k-nearest neighbors model
knn = KNeighborsClassifier()
knn.fit(x_tr,y_tr)
pred = knn.predict(x_ts)
print('Knn',classification_report(y_ts,pred))



