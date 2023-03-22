import numpy as np
import matplotlib.pyplot as plt

# %%
data_size = 500
x_range = 8
function_real = lambda x: 2 * x * x - 14 * x + 24
noise = 2

epoches = 100000
learning_rate = 0.001
# %%
x_data = np.random.rand(data_size) * x_range
y_data = np.array(list(map(function_real, x_data))) + np.random.randn(data_size) * 2 * noise - noise
# %%
a, b, c = 1, 1, 1
function_pred = lambda x: a * x * x + b * x + c
# %%
for i in range(epoches):
    prediction = function_pred(x_data)
    a -= sum(2 * (prediction - y_data) * x_data * x_data) / data_size * learning_rate
    b -= sum(2 * (prediction - y_data) * x_data) / data_size * learning_rate
    c -= sum(2 * (prediction - y_data)) / data_size * learning_rate

    print(f"epoch{i} : {(sum((prediction - y_data) ** 2) / data_size)}")
# %%
plt.scatter(x_data, y_data)
plt.plot(range(x_range + 1), list(map(function_real, range(x_range + 1))), color='red')
plt.plot(range(x_range + 1), list(map(function_pred, range(x_range + 1))), color='orange')
plt.show()
# %%
print(a, b, c)
# %%
