import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Training dataset
data = {
    "Temperature": [20, 21, 22, 23, 24, 25, 26, 27],
    "Salinity": [30, 31, 29, 32, 33, 34, 30, 29],
    "Count": [5, 6, 7, 9, 10, 12, 14, 15]
}
df = pd.DataFrame(data)

X = df[["Temperature"]]   # just one feature for visualization
y = df["Count"]

model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Temperature (°C)")
plt.ylabel("Fish Count")
plt.title("Fish Count vs Temperature")
plt.legend()
plt.show()
