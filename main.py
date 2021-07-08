#Code by anmabe06
#May 13th of 2021


#Import libraries
import pandas #To read certain files
#To make complex math
import numpy as np
#To make training
from sklearn.model_selection import train_test_split
#Import nearnest neighbour algorithm
from sklearn.neighbors import KNeighborsClassifier
#To make model accuracy report
from sklearn import metrics


#To show graph
import seaborn as sns
import matplotlib.pyplot as plt


#Read dataset
iris = pandas.read_csv("/content/IRIS.csv")


#Separate data into input and exit variables
y = np.array(iris['species']) #Exit
x = np.array(iris.drop(['species'], 1)) #Input


#Separate data into test and train values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


#Apply KNN algorith
neighbours = 3
model = KNeighborsClassifier(n_neighbors=neighbours)
model.fit(x_train,y_train)


#Train model and search for predictions
y_pred = model.predict(x_test)

#Set scores
scores= metrics.accuracy_score(y_test, y_pred)


#Print
print('Model precision: {}%'.format(scores*100))


#Show graph
graph = sns.pairplot(iris, hue='species', markers='+')
plt.show()
