#program_2
#Model regresji liniowej służący do przewidywania oceny na podstawie czasu nauki, x-liczba godzin poświęconych na naukę,

yy to przewidywana ocena.



xx to liczba godzin poświęconych na naukę,

yy to przewidywana ocena

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.arange(1, 11)
x = x.reshape(-1, 1)
np.random.seed(0)
y = 5 * x + 50 + np.random.normal(0, 5, size=x.shape)

model_regresja = LinearRegression()
model_regresja.fit(x, y)

print("Współczynnik kierunkowy:", model_regresja.coef_)
print("Wyraz wolny:", model_regresja.intercept_)

y_pred = model_regresja.predict(x)

plt.scatter(x, y, color='green', label='Dane')  
plt.plot(x, y_pred, color='red', label='Regresja liniowa')  
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()