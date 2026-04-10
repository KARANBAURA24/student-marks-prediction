import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'Hours': [1,2,3,4,5,6,7,8],
    'Marks': [10,20,30,40,50,60,70,80]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Marks']

model = LinearRegression()
model.fit(X, y)

pred = model.predict([[5.5]])
print("Predicted Marks:", pred[0])

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()