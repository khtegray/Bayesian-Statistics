# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 18:08:00 2026

@author: khate
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Create a tiny dataset (Years of Experience vs Salary)
# X needs to be a 2D array for sklearn
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]]) 
y = np.array([45000, 50000, 60000, 80000, 110000, 150000, 190000, 230000])

# 2. Initialize the Linear Regression model
model = LinearRegression()

# 3. Train the model
model.fit(X, y)

# 4. Predict the salary for someone with 10 years of experience
years_new = [[10]]
predicted_salary = model.predict(years_new)

print(f"Predicted salary for 10 years experience: ₱{predicted_salary[0]:,.2f}")

# 5. Visualizing the "Best Fit Line"
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()