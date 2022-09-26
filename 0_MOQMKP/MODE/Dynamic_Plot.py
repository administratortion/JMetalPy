"""
by zhanxin 2022.1
"""


import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


# 归一化函数
def Normalization(data):
    min1 = min(data)
    max1 = max(data)
    for i in range(len(data)):
        data[i] = (data[i] - min1) / (max1 - min1)
    print("min: ", min1, "max: ", max1)
    # print(data)


# 画三维描点图
def show(X, Y, Z):
    ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
    ax.set_title('3d_image_show')  # 设置本图名称
    ax.set_xlabel('X')  # 设置x坐标轴
    ax.set_ylabel('Y')  # 设置y坐标轴
    ax.set_zlabel('Z')  # 设置z坐标轴
    for j in range(len(X) - 1):
        x1 = np.array(X[j])
        y1 = np.array(Y[j])
        z1 = np.array(Z[j])
        ax.scatter(x1, y1, z1)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
        plt.pause(2)  # 播放速度

    ax.set_xlabel('X')  # 设置x坐标轴
    ax.set_ylabel('Y')  # 设置y坐标轴
    ax.set_zlabel('Z')  # 设置z坐标轴
    plt.show()


# 打开文件读入数据
def read(fileName):
    X = []
    Y = []
    Z = []
    serial_n = []
    tmp = 0
    with open(fileName) as file:
        lines = file.readlines()
        data = [line.split() for line in lines if len(line.split()) >= 1]
        for data1 in data:
            if '---' in data1[0]:
                serial_n.append(tmp)
                tmp = 0
                continue
            X.append(int(float(data1[0])))  # 字符串转换为int 类型
            Y.append(int(float(data1[1])))
            Z.append(int(float(data1[2])))
            tmp = tmp + 1
    file.close()
    return X, Y, Z, serial_n


# 读取颜色名称
def readColor(fileName):
    with open(fileName) as file:
        color = []
        lines = file.readlines()
        data = [line.split() for line in lines if len(line.split()) >= 1]
        for data1 in data:
            color.append(data1[0])
    return color


# 按代数分割文件读取后的原始数据列表
def division(serial_n, x1):
    X = []
    j = 0
    i = 0
    for j in serial_n:
        X.append(x1[i:i + j])
        i = i + j
    return X


# 按每代的数量分割归一化处理后的数据
def y_axis_data(x1, y1, z1):
    y_axis = []
    y_axis2 = []
    for i in range(len(x1)):
        y_axis.append(x1[i])
        y_axis.append(y1[i])
        y_axis.append(z1[i])
        y_axis2.append(y_axis)
        y_axis = []
    return y_axis2


# 折线图
def draw_pic_test(mat, color):
    month_list = [0, 1, 2]
    for one_list in mat:
        color1 = random.sample(color, 1)
        for tmp in one_list:
            x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
            ax = plt.gca()  # ax为两条坐标轴的实例
            ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
            plt.plot(month_list, tmp, color1[0], label="test_zhexian")
        plt.pause(1)  # 播放速度
    plt.show()
    plt.close()


if __name__ == '__main__':
    # 读入数据
    # serial_n存储每一代的数量
    # x, y, z, serial_n = read('../NSGAII/FUN.NSGAII.MOQMKP_PER')
    x, y, z, serial_n = read('FUN.MODE.MOQMKP_PER')
    # color = readColor('color1.txt')

    # 归一化处理
    print("X处理：")
    Normalization(x)
    print("Y处理：")
    Normalization(y)
    print("Z处理：")
    Normalization(z)

    # 动态画折线图
    # y2 = y_axis_data(x, y, z)
    # y3 = division(serial_n, y2)
    # draw_pic_test(y3, color)  # 动态画法

    # 动态画散点图
    x = division(serial_n, x)
    y = division(serial_n, y)
    z = division(serial_n, z)
    show(x, y, z)   # 三维图描点
