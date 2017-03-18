from statistics import mean

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#x_set = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
#y_set = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


# plt.plot(x_set, y_set) # plot with line
# plt.scatter(x_set, y_set) #plot with dots
# plt.show()

# Testing assumptions
def create_dataset(datasize, variance, step=2, correlation=False):
    val = 1
    y_set = []
    for i in range(datasize):
        y = val + random.randrange(-variance, variance)
        y_set.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    x_set = [i for i in range(len(y_set))] # from 1 to datasize of y

    return np.array(x_set, dtype=np.float64), np.array(y_set, dtype=np.float64)


def best_fit_slope_and_intercept(x_set, y_set):
    m = (((mean(x_set) * mean(y_set) - mean(x_set * y_set)) /
          (mean(x_set) ** 2 - mean(x_set ** 2))))
    # mean(x_set)**2 # = math.sqrt(mean(x_set))
    b = mean(y_set) - m * mean(x_set)
    return m, b


def squared_error(y_set_original, regression_line):
    return sum((regression_line - y_set_original) ** 2)


def coefficient_of_determination(y_set_original, regression_line):
    y_mean_line = [mean(y_set_original) for y in y_set_original]
    squared_error_regressionline = squared_error(y_set_original, regression_line)  # y headline
    squared_error_y_mean = squared_error(y_set_original, y_mean_line)
    return 1 - (squared_error_regressionline / squared_error_y_mean)

x_set, y_set = create_dataset(40, 80, 2, correlation=False)
m, b = best_fit_slope_and_intercept(x_set, y_set)

regression_line = [(m * x) + b for x in x_set]  # create points in regression line

# instead of lambda, we may have following solution for the above for loop
# for x in x_set:
#     regression_line.append((m * x) + b)

given_x = 8
predict_y = (m * given_x) + b

r_squarred = coefficient_of_determination(y_set, regression_line)
print(r_squarred)

plt.scatter(x_set, y_set)
#plt.scatter(given_x, predict_y, color='g')
plt.plot(x_set, regression_line)
plt.show()
