from enum import Enum
from AnalysisToolkit_LSH.MOQMKP_Package.dataset import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from jmetal.util.observer import Observer
import numpy as np


class CoordinateType(Enum):  # 未使用
    XY = 0
    XZ = 1
    YZ = 2
    XYZ = 3


class Plotting(Observer):
    def __init__(self):
        # 颜色
        self.target_color = 'r'  # 目标颜色为红色
        self.associated_color = 'g'  # 与目标关联的为绿色

        # 数据集
        # 相关格式，请自行查看 documents\plotting_docs\datasetInitFormat.png
        self.receive_dataset = None  # 数据集
        # 获得interval间隔内的所有种群的objectives（三维）点集，分成 x, y, z,请看format_data函数
        self.plot_data: dict = {}

        # 画图
        self.fig = plt.figure()
        # self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ax1 = self.fig.add_subplot(111, projection='3d')

    # self.ax2 = self.fig.add_subplot(122, projections='3d')

    def format_data(self):
        """
        通过 np.array().T[] 快速获取二维列表的某一列所有元素
        Returns:
            返回已经格式化的数据
        """

        i = 1  # 指第几个参考点
        plot_data: dict = {}
        for target, associated in self.receive_dataset:
            # 数据格式化
            final_x = final_y = final_z = []
            final_x = target[0]
            final_y = target[1]
            final_z = target[2]
            plot_data[i] = [[final_x], [final_y], [final_z]]

            if len(associated) != 0:
                plot_data[i][0].extend(np.array(associated).T[0])
                plot_data[i][1].extend(np.array(associated).T[1])
                plot_data[i][2].extend(np.array(associated).T[2])

            i = i + 1
        self.plot_data = plot_data

    def plotting_from_file(self, filename):
        dataset = []
        with open(filename, 'r+', encoding='utf-8') as f:
            while True:
                file_line = f.readline()
                if not file_line:
                    break
                dataset.append(eval(file_line))

        i = 0  # 当前代
        for item in dataset:
            self.receive_dataset = item
            self.format_data()

            self.ax1.cla()  # 清空左侧子图数据
            for key, value in self.plot_data.items():  # key参考点的坐标
                target_x, target_y, target_z = value[0][0], value[1][0], value[2][0]
                associated_x, associated_y, associated_z = value[0][1:], value[1][1:], value[2][1:]
                self.ax1.scatter3D(target_x, target_y, target_z, color=self.target_color, marker='x')  # target
                self.ax1.scatter3D(associated_x, associated_y, associated_z, color=self.associated_color)  # associated
                self.ax1.set_title(f'{i} generations ----- {key} reference_pointer')

            plt.draw()  # 重绘函数
            plt.pause(1)
            i = i + 1

    def plotting(self):
        self.format_data()
        self.plot_3d()

    def plot_3d(self):
        self.ax1.cla()  # 清空左侧子图数据
        for key, value in self.plot_data.items():
            target_x, target_y, target_z = value[0][0], value[1][0], value[2][0]
            associated_x, associated_y, associated_z = value[0][1:], value[1][1:], value[2][1:]
            self.ax1.scatter3D(target_x, target_y, target_z, color=self.target_color, marker='x')  # target
            self.ax1.scatter3D(associated_x, associated_y, associated_z, color=self.associated_color)  # associated
            self.ax1.set_title('3D objectives plot')
            plt.draw()
            plt.pause(1)
        # plt.pause(0.8)

    # plt.show()
    # plt.close()

    def plot_2d(self):
        pass

    def update(self, *args, **kwargs):
        tmp: Dataset = kwargs["dataset"]
        self.receive_dataset = tmp.plotting_objectives_dataset
        self.plotting()

# def on_key_press(self, event):
# 	if event.key == ' ':
# 		plt.pause(6)
