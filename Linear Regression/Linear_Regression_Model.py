import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Random  data
rng = np.random

#การจำลองข้อมูล
x = (rng.rand(50))*10 #rand + number (no -)
y = 2*x + rng.randn(50) #rand (+,-) number

# 2D array
x_new = x.reshape(-1,1)

#Linear regression model;
model = LinearRegression()

#Train model
model.fit(x_new,y) 
print(model.score(x_new,y)) #R-squre วัดการตัดสินใจในการเลือก y 0-100 %
print(model.coef_) #coefficient

#Test model
xfit = np.linspace(-1,11)
xfit_new = xfit.reshape(-1,1)
print(xfit_new.shape) # (50,1)

yfit = model.predict(xfit_new)

#Analysis model/Result
plt.scatter(x,y)
plt.plot(xfit,yfit)
plt.show()