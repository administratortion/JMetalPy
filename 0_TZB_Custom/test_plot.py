import random
variables = [1, 2, 3, 4, 5, 6, 7]
point1=1
point2=5
values = variables[point1:point2]
variables[point1:point2] = random.sample(values, len(values))

a = 1

"""
二维散点图
"""
"""
import numpy as np
import matplotlib.pyplot as plt

N=40
x=np.random.rand(N)
y=np.random.rand(N)*10

# random colour for points, vector of length N
colors=np.random.rand(N)

# area of the circle, vectoe of length N
area=(30*np.random.rand(N))**2
# 0 to 15 point radii
"""

"""
# a normal scatter plot with default features
plt.scatter(x, y, alpha=0.8)
plt.xlabel('Numbers')
plt.ylabel('Values')
plt.title('Normal Scatter Plot')
plt.show()

# a scater plot with different size
plt.figure()
plt.scatter(x, y, s=area, alpha=0.8)
plt.xlabel('Numbers')
plt.ylabel('Values')
plt.title('Different Size')
plt.show()

# a scatter plot with different collour
plt.figure()
plt.scatter(x, y, c=colors, alpha=0.8)
plt.xlabel('Numbers')
plt.ylabel('Values')
plt.title('Different Colour')
plt.show()

# A combined Scatter Plot
plt.figure()
plt.scatter(x, y, s=10, c=colors, alpha=0.8)
plt.xlabel('Numbers')
plt.ylabel('Values')
plt.title('Combined')
plt.show()
"""

'''
三维散点图
'''
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

x = np.array([1, 2, 4, 5, 6])
y = np.array([2, 3, 4, 5, 6])
z = np.array([1, 2, 4, 5, 6])

ax = plt.subplot(projection = '3d')  # 创建一个三维的绘图工程
ax.set_title('3d_image_show')  # 设置本图名称
ax.scatter(x, y, z, c = 'r')   # 绘制数据点 c: 'r'红色，'y'黄色，等颜色

ax.set_xlabel('X')  # 设置x坐标轴
ax.set_ylabel('Y')  # 设置y坐标轴
ax.set_zlabel('Z')  # 设置z坐标轴

plt.show()
"""


"""
归一化
"""
"""
data = [-1, -2, -3, -4, -5]
min=min(data)
max=max(data)
for i in range(len(data)):
    data[i] = (data[i]-min)/(max-min)
print("min: ", min, "max: ", max)
print(data)
"""


