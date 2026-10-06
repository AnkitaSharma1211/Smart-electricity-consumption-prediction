import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#Load the dataset
data = pd.read_csv("project1.py/electricity_data.csv")

print("First 5 rows:")
print(data.head())


# Check the dataset
print("\nDataset information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())


# Select input and output
X = data[["Temperature", "Humidity", "Previous_Usage", "Hour"]]

y = data["Consumption"]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create the ML model
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Check model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-----------------")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# Predict electricity consumption
temperature = float(input("Enter temperature: "))
humidity = float(input("Enter humidity: "))
previous_usage = float(input("Enter previous electricity usage: "))
hour = int(input("Enter hour (0-23): "))

new_data = pd.DataFrame({
    "Temperature": [temperature],
    "Humidity": [humidity],
    "Previous_Usage": [previous_usage],
    "Hour": [hour]
})

prediction = model.predict(new_data)

print("\nPredicted Electricity Consumption:")
print(round(prediction[0], 2), "kWh")


# Make a graph
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Consumption")
plt.ylabel("Predicted Consumption")
plt.title("Actual vs Predicted Electricity Consumption")

plt.savefig("graph.png")
plt.show()