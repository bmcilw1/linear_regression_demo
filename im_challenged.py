import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
 
# read data
dataframe = pd.read_csv('challenge_dataset.txt')
x_values = dataframe[[0]]
y_values = dataframe[[1]]
test_val = (dataframe.iloc[20][0], dataframe.iloc[20][1])

print(test_val)

print("Fitting...")

# train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

print("Done!")

predict_val = body_reg.predict(test_val[0])

print("Sample: %f, %f" % (test_val[0], test_val[1]))
print("Predict: %f, %f" % (test_val[0], predict_val))
print("Error: %i%%" % ((test_val[1] - predict_val) / test_val[1] * 100))

# visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.scatter(test_val[0], predict_val, color='red', marker='o')
plt.scatter(test_val[0], test_val[1], color='red', marker='+')
plt.show()
