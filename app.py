import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

titanic = sns.load_dataset("titanic")
df_titanic = titanic[['sex', 'age', 'survived']].copy()
df_titanic['age'].fillna(df_titanic['age'].mean(), inplace=True)
df_titanic['sex'] = df_titanic['sex'].map({'male': 0, 'female': 1})
print(df_titanic.head())

df_titanic['survived'] = df_titanic['survived'].astype(int)
x = df_titanic[['sex', 'age']]
y = df_titanic['survived'] 
print(df_titanic['survived'].unique()) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape)

model = Perceptron(max_iter=1000, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładość modelu", accuracy)

print(y_pred[:10])

new_data = pd.DataFrame([[0, 25], [1, 30]], columns=['sex', 'age'])
predictions = model.predict(new_data)
print("Przewidywania dla nowych danych", predictions)