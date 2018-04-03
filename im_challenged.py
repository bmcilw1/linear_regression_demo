import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
 
# read data
dataframe = pd.read_csv('challenge_dataset.txt')
x_values = dataframe[[0]]
y_values = dataframe[[1]]
test_val = (dataframe[[0]].iloc[0], dataframe[[1]].iloc[0])

#print(test_val)

print("Fitting...")

# train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

print("Done!")

print("Sample: %i, %f" % (test_val, body_reg.predict(test_val)))

# visualize results
'''
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.scatter(test_val, body_reg.predict(test_val))
plt.show()
'''
