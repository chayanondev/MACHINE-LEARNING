import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

dataset = pd.read_csv("https://raw.githubusercontent.com/kongruksiamza/MachineLearning/master/Linear%20Regression/Weather.csv")
#data (119040,31)

#Plot
'''
dataset.plot(x='MinTemp',y='MaxTemp',style="o")
plt.title('Min and Max Temperature')
plt.xlabel("Min Temperature")
plt.ylabel("Max Temperature")
plt.show()
'''

#Train and Test set 
x = dataset["MinTemp"].values.reshape(-1,1)
y = dataset["MaxTemp"].values.reshape(-1,1)

#Split data 80% and 20%
x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=0
)

#Training
model = LinearRegression()
model.fit(x_train,y_train)

#Test 
y_predict = model.predict(x_test)

plt.scatter(x_test,y_test) 
plt.plot(x_test,y_predict,color='red',linewidth=2)#แยกข้อมูลบน-ล่าง
plt.show()

# Compare true data and predict data (To 1D)
df = pd.DataFrame({'Actually':y_test.flatten(),"Predicted":y_predict.flatten()})
#(23808, 2)

df1 = df.head(20)
df1.plot(kind="bar",figsize=(16,10))
plt.show()


#Coefficient
print("MAE = ",metrics.mean_absolute_error(y_test,y_predict))
print("MSE = ",metrics.mean_squared_error(y_test,y_predict))
print("RMSE = ",np.sqrt(metrics.mean_absolute_error(y_test,y_predict)))
print("Score = ",metrics.r2_score(y_test,y_predict))