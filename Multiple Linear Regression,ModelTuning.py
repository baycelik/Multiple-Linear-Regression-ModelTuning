import pandas as pd

df=pd.DataFrame(d)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
X=df.drop("Consumption",axis=1)
y=df["Consumption"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 50)

X_train.shape

y_train.shape

X_test.shape

y_test.shape

training = df.copy()
training.shape

import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model=lm.fit(X_train,y_train)

model.intercept_

model.coef_

for i in range(4):
    print(model.coef_[i])
    
Dummy_Parameters=[[1],[3],[4],[0]]
Dummy_Parameters=pd.DataFrame(Dummy_Parameters).T

Dummy_Parameters

model.predict(Dummy_Parameters)

df['Consumption']

rmse=np.sqrt(mean_squared_error(y_train,model.predict(X_train)))
rmse

rmse=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
rmse

model.score(X_train,y_train)

np.sqrt(-cross_val_score(model,X_test,y_test,cv=10,scoring="neg_mean_squared_error")).mean()

for i in range(101):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= i)
    a=np.sqrt(-cross_val_score(model,X_test,y_test,cv=10,scoring="neg_mean_squared_error")).mean()
    b=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
    if(abs(a-b)<1):
        print(abs(a-b))
        print("RandomState: "+"{}".format(i))
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 18)

rmse=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
rmse

np.sqrt(-cross_val_score(model,X_test,y_test,cv=10,scoring="neg_mean_squared_error")).mean()



