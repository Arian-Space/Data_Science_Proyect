# Import libraries for the code
import pandas as pd # Taking like "pd"
import matplotlib.pyplot as plt # Taking like "plt"
from sklearn.linear_model import LinearRegression

# Use Pandas to import the data from epa-sea-level.csv
orders = pd.read_csv('https://pkgstore.datahub.io/core/sea-level-rise/csiro_alt_gmsl_mo_2015_csv/data/dc258c2039d8b640f74efd3d23e1c920/csiro_alt_gmsl_mo_2015_csv.csv') # Read the (.csv)

# Data for the axis (for resume)
x_orders = (orders['Time']).tolist() # x data
y_orders = (orders['GMSL']).tolist() # y data

# Index for searching
indexes_x = 1993 # Year for x
j = 0 # Index for y data

# Define the new arrays
new_orders_x = [] # For the graph (x-scatterplot)
mean_y = [] # For mean
new_orders_y = [] # For the graph (y-scatterplot)

# Cicle to simplify data (x axis to int and mean for y)
for i in x_orders:

  if str(indexes_x) == i[0:4]:
    # For x
    if str(indexes_x) not in str(new_orders_x):
      new_orders_x.append(indexes_x)

    # For y
    mean_y.append(y_orders[j])

  else: # Next year
    indexes_x += 1 # Next year

    # For x
    new_orders_x.append(indexes_x) # Happy new year!!!!

    # For y
    result = round((pd.Series(mean_y)).mean(),2) # Mean for data
    new_orders_y.append(result) # Take the result for graph
    mean_y = [] # Taking out the data
    mean_y.append(y_orders[j]) # Taking the first data of the next year

  j += 1 # Index for the data

result = round((pd.Series(mean_y)).mean(),2) # Last mean for data
new_orders_y.append(result) # Last data include in y

# Checking the Data
print('Data for the x-axis (unique years):') # For user from x
print(new_orders_x) # List for data in x
print() # Enter
print('Data for the y-axis (mean for each year):') # For user from y
print(new_orders_y) # List for data in y
print() # Enter

# Creating a linear regression model
model = LinearRegression()

# Fit the model to our data
model.fit(pd.DataFrame(new_orders_x), pd.DataFrame(new_orders_y))

# Prediction for the year 2050
prediction_2050 = model.predict([[2050]])[0][0]
# Prediting data form the function

# Print the prediction
print("Prediction for 2050:", round(prediction_2050, 2))
print() # Enter

# Built the scatterplot
plt.scatter(new_orders_x, new_orders_y) # Dump in data

# Titles for the x and y axes
plt.xlabel('Year') # x-axis
plt.ylabel('CSIRO Adjusted Sea Level in (mm)') # y-axis

# Add a title to the chart
plt.title('Rise in Sea Level') # Title

# Regression line (Ploting the line)
plt.plot(new_orders_x, model.predict(pd.DataFrame(new_orders_x)), color='red')

# show graph (scatterplot)
plt.show() # Go!!!
