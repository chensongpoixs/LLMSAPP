# NumPy 
# A fundamental package for scientific computing with Python.
from locale import ABDAY_1
import numpy as np;

# Pandas
# A powerful data manipulation and analysis library for Python.
import pandas as pd;


# nmpy 一维数组
a = np.array([1, 2, 3, 4, 5]);
print("NumPy 一维数组:", a);
# pandas 数据框
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]};
df = pd.DataFrame(data);
print("Pandas 数据框:\n", df);



# Matplotlib
# A comprehensive library for creating static, animated, and interactive visualizations in Python.
# numpy 二维数

a = np.array([[1, 2, 3], [4, 5, 6]]);
print("NumPy 二维数组:\n", a);

# numpy 三维数组
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
print("NumPy 三维数组:\n", a);

g = np.array([[1, 2, 3], [4, 5, 6]]);
h = np.array([[7, 8, 9], [10, 11, 12]]);
# 矩阵加法
sum_result = g + h;
print("矩阵加法结果:\n", sum_result);
# 矩阵乘法
product_result = np.dot(g, h.T);
print("矩阵乘法结果:\n", product_result);

import matplotlib.pyplot as plt;
# 创建数据
x = np.linspace(0, 10, 100);
y = np.sin(x);
# 创建图形
plt.plot(x, y);
    