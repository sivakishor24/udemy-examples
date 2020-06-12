#part 1

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load the dataset
df = pd.read_csv("C:/Users/Dell/Desktop/datasets/Churn_Modelling.csv")
#print(df.head())

#preprocessing the dataset
'''print(df.columns)
print(df.values)
print(df.size)
print(df.shape)
print(df.duplicated())
print(df.isna())'''

#we have to split the independent and dependent columns
X = df.iloc[:, 3:13]
Y = df.iloc[:, 13]
#print(X.shape)
#print(Y.shape)

#print(df['Geography'])
#print(df['Gender'])

#create dummy variables for two columns (geography & gender)
geography = pd.get_dummies(X['Geography'],drop_first=True)
gender = pd.get_dummies(X['Gender'],drop_first=True)
#print(geography)
#print(gender)

#concatenate the dataframe
X = pd.concat([X,geography,gender],axis=1)
#print(X)

#drop the unnecessary columns
X = X.drop(['Geography','Gender'],axis = 1)
#print(X)

#splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.25, random_state=0)
'''print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)'''

#feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
#print(X_train)
#print(X_test)

#part 2----Now lets make the ANN
import keras
from keras.models import Sequential #It is responsible for creatingany neural network(CNN, RNN, ANN)
from keras.layers import Dense #using for hidden layers
#from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout #using for dropout the unwanted layers basically it is a regularization parameter
import warnings
warnings.filterwarnings('ignore')

#Initialising the ANN
classifier = Sequential()

#Adding theinput layer and the first hidden layer
#classifier.add(Dense(output_dim = 6, init = 'he_uniform', activation='relu', input_dim = 11))
#It throws some warning
classifier.add(Dense(units = 10, kernel_initializer = 'he_normal', activation='relu', input_dim = 11))
#Adding dropout layer
classifier.add(Dropout(0.3))

#Adding the second hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation = 'relu'))
classifier.add(Dense(units = 20, kernel_initializer = 'he_normal', activation = 'relu'))
#Adding dropout layer
classifier.add(Dropout(0.4))

#Adding the third hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'he_normal', activation = 'relu'))
#Adding dropout layer
classifier.add(Dropout(0.2))

#Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training set
model_history = classifier.fit(X_train, Y_train, validation_split = 0.25, batch_size = 10, nb_epoch=10)

#list all data in history
#print(model_history.history.keys())

#summarize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'],loc='upper left')
plt.show()

#part 3--Making the predictions and evaluting the model

#predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #here the y_pred value is less than 0.5 is 'False' and greater than 0.5 is 'True'
print(y_pred)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, y_pred))

#calculating the Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, Y_test))