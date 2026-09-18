import numpy as np
import pandas as pd

# import os, glob
# print(os.listdir("/kaggle/input"))
# print(glob.glob("/kaggle/input/**/blood_count_dataset.csv", recursive=True))


# Load the CBC dataset
df = pd.read_csv(
  "C:\Users\Pankaj Krishna\Desktop\ML_5thSem\Blood Cell Counting Dataset (BCCD).csv")

# 70% training and 30% testing
n = len(df)
train_size = int(0.7 * n)

train_set = df[:train_size]
test_set = df[train_size:].copy()


# Manual median function
def median_value(values):
    values = sorted(values)
    n = len(values)

    if n % 2 == 0:
        return (values[n // 2 - 1] + values[n // 2]) / 2
    else:
        return values[n // 2]


# Extract x and y from training data
hb_train = list(train_set["Hemoglobin"])
wbc_train = list(train_set["White_Blood_Cells"])

# Median of all training samples
median_hb = median_value(hb_train)

# Divide training samples using median Hemoglobin
p1 = train_set[train_set["Hemoglobin"] <= median_hb]
p2 = train_set[train_set["Hemoglobin"] > median_hb]

# Two representative median points
x1 = median_value(list(p1["Hemoglobin"]))
y1 = median_value(list(p1["White_Blood_Cells"]))

x2 = median_value(list(p2["Hemoglobin"]))
y2 = median_value(list(p2["White_Blood_Cells"]))

print("Point 1:", x1, y1)
print("Point 2:", x2, y2)

# Fit y = mx + c manually
m = (y2 - y1) / (x2 - x1)
c = y1 - m * x1

print("Slope:", m)
print("Intercept:", c)
print(f"Line equation: WBC = {m:.4f} * Hemoglobin + {c:.4f}")


# Prediction function
def predict(x):
    return m * x + c


# Predict WBC for test samples
test_set["Predicted_WBC"] = test_set["Hemoglobin"].apply(predict)

print(test_set[
    ["Hemoglobin", "White_Blood_Cells", "Predicted_WBC"]
])


# Calculate MAE, MSE and RMSE manually
actual = list(test_set["White_Blood_Cells"])
predicted = list(test_set["Predicted_WBC"])

total_abs_error = 0
total_squared_error = 0

for i in range(len(actual)):
    error = actual[i] - predicted[i]
    total_abs_error += abs(error)
    total_squared_error += error * error

mae = total_abs_error / len(actual)
mse = total_squared_error / len(actual)
rmse = mse ** 0.5

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

