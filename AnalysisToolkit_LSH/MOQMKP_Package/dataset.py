"""
用途：从进化算法过程中收集历史数据，用于神经网络训练，
开发：森辉
时间：2022.4
"""

import copy
import numpy as np
from jmetal.config import store
from jmetal.core.problem import Problem
from jmetal.util.comparator import Comparator
from AnalysisToolkit_LSH.MOQMKP_Package.compare import NondominanceCompare


class ASF:
    """
        best_asf : 生成与跟reference_points.shape一样的相关联的target (决策向量)

        ref_asf : (target_variables, associated_variables)，
                    其中target_variables跟参考点关联(即best_asf生成的target)，associated_variables跟target_variables关联
    """
    @staticmethod
    def best_asf(reference_points, target_objectives, target_variables):
        """
        设 T 为 target与reference_points关联的列表
        返回 T
        算法：遍历所有参考点，每个参考点遍历所有当前父代个体
        
        Args:
            reference_points:   参考点
            target_objectives:  解的目标空间
            target_variables:   解的决策空间

        Returns:
            target: 与reference_points关联的列表     跟参考点是的类型是一样的
            other_objectives_variables_pair:  =  输入的 - 输出的
        """
        target_objectives_variables_pair = list(zip(copy.deepcopy(target_objectives), copy.deepcopy(target_variables)))
        other_objectives_variables_pair = copy.deepcopy(target_objectives_variables_pair)  # 除 target 外的多余解集

        number_of_reference_points = len(reference_points)
        number_of_target = len(target_objectives_variables_pair)

        target = [[] for _ in range(len(reference_points))]  # return target <- 跟参考点关联的解集

        for reference_index in range(number_of_reference_points):
            tmp = []
            for target_index in range(number_of_target):
                tmp.append([tar_obj - ref for tar_obj, ref in
                            zip(target_objectives_variables_pair[target_index][0], reference_points[reference_index])])
            tmp = [max(every_element) for every_element in tmp]

            min_index = tmp.index(min(tmp))
            target[reference_index] = target_objectives_variables_pair[min_index][1]

            # 如果差解中有该轮选出的目标解，则从差解集合中剔除
            if target_objectives_variables_pair[min_index] in other_objectives_variables_pair:
                other_objectives_variables_pair.remove(target_objectives_variables_pair[min_index])

        return target, other_objectives_variables_pair

    @staticmethod
    def ref_asf(reference_points, associated_objectives, associated_variables, target_variables):
        """
        设 dataset 为 一个列表，列表为 (target_variables, associated_variables)
        算法：遍历差解，每个差解遍历参考点，根据ASF绑定差解和参考点
        Args:
            reference_points:       参考点
            associated_objectives:  与target关联的目标空间
            associated_variables:   与target关联的决策空间
            target_variables:       best_asf生成的target(已经跟参考点关联了)

        Returns:
            返回列表 (target_variables, associated_variables)，
            其中target_variables跟参考点关联，associated_variables跟target_variables关联
        """
        associated_objectives_variables_pair = list(
            zip(copy.deepcopy(associated_objectives), copy.deepcopy(associated_variables)))
        number_of_reference_points = len(reference_points)
        number_of_associated = len(associated_objectives_variables_pair)

        associated_output = [[] for _ in range(number_of_reference_points)]

        for associated_index in range(number_of_associated):
            tmp = []
            for reference_index in range(number_of_reference_points):
                tmp.append([associated_obj - ref for associated_obj, ref in
                            zip(associated_objectives_variables_pair[associated_index][0],
                                reference_points[reference_index])])

            tmp = [max(every_element) for every_element in tmp]
            min_index = tmp.index(min(tmp))
            associated_output[min_index].append(associated_objectives_variables_pair[associated_index][1])

        dataset = list(zip(target_variables, associated_output))
        return dataset


class Normal:
    """
        归一化
    """
    @staticmethod
    def normalized(F, ideal_point, nadir_point):
        """
        分子：目标向量 - 理想点
        分母：最差点 - 理想点
        _F = 分子 / 分母
        Args:
            F:             原始目标向量空间
            ideal_point:   理想点
            nadir_point:   最差点
        Returns:  F归一化，使_F落在在[0,1]
        """
        if len(F[0]) != len(ideal_point):
            raise Exception('Dimensionality of F must be equal to the number of ideal_point in Normalization')

        normalized_F = copy.deepcopy(F)
        denominator = list([i - j for i, j in zip(nadir_point, ideal_point)])  # 分母相减
        for index in range(len(normalized_F)):
            normalized_F[index] = list([i - j for i, j in zip(normalized_F[index], ideal_point)])  # 分子相减
            normalized_F[index] = list([i / j for i, j in zip(normalized_F[index], denominator)])  # 分子 / 分母
        return normalized_F


class Dataset:
    """
        生成各种数据集
    """
    def __init__(self,
                 reference_points,
                 problem: Problem,
                 dominance_comparator: Comparator = store.default_comparator):
        self.reference_points = reference_points.compute()          # 产生参考点的方法
        self.problem = problem                                      # problem
        self.dominance_comparator = dominance_comparator            # 非支配比较算子

        self.ideal_point = np.full(self.problem.number_of_objectives, np.inf)   # 理想点
        self.nadir_point = np.full(self.problem.number_of_objectives, -np.inf)  # 最差点

        self.datasetToNNetwork = []                                 # 产生适用于神经网络的数据集
        self.all_populations = []                                   # 所有解集
        self.all_without_current_populations = []                   # 不包含当前代的所有解集
        self.current_generations = 0                                # 当前第几代
        self.target = None                                          # target.shape == reference.shape，它与参考点相关联
        self.associated = None                                      # 在(interval)中除去target的剩余的解集
        self.with_compare_result_datasetToNNetwork = None           # 在datasetToNNetwork基础上带有比较结果
        self.plotting_objectives_dataset = []                       # 用于plotting使用的数据集

        self.non_compare = NondominanceCompare(self.problem)        # 具有比较结果的实例
        self.observable = store.default_observable                  # 观察者模式中的被观察者

    def run_by_setting_interval(self, interval):
        """
            核心方法，用于 notify 以及进行相关计算
        Args:
            interval:  间隔，将间隔内的所有种群进行计算

        Returns:

        """
        if len(self.all_populations) >= interval:
            self.get_nnetwork_dataset(interval)                             # 数据计算
            self.get_with_compared_result_from_datasetToNNetwork()          # 比较修复前后子代情况
            self.get_plotting_objectives_dataset()                          # 获取绘图数据

            self.observable.notify_all(**(self.get_observable_data()))      # 通知观察者

    def add_current_population(self, all_population):
        self.all_populations = all_population

    def get_nnetwork_dataset(self, interval):
        """
            生成训练神经网络的数据集
        Args:
            interval: 一共interval个间隔内的所有种群数，生成用于神经网络的数据集

        Returns:
            结果保存到self.datasetToNNetwork中，关联集
        """
        # 当前第几代
        self.current_generations = len(self.all_populations)

        # 一共 (interval - 1) 代但不包含当前代
        self.all_without_current_populations = copy.deepcopy(self.all_populations[-interval: -1])
        # 二维列表转换成一维列表
        self.all_without_current_populations = [n for x in self.all_without_current_populations for n in x]
        # 当前代
        current_population = copy.deepcopy(self.all_populations[-1])

        # 备份目标值
        # 归档集但不包含当前代的（interval - 1）代
        achieve_objectives = [s.objectives for s in self.all_without_current_populations]
        # 当前目标值
        current_objectives = [s.objectives for s in current_population]

        # 备份变量
        achieve_variables = [s.variables for s in self.all_without_current_populations]
        current_variables = [s.variables for s in current_population]

        # 从全体解集计算出理想点和极端点
        all_objectives = copy.deepcopy(self.all_populations[-interval:])
        all_objectives = [n for x in all_objectives for n in x]
        all_objectives = [s.objectives for s in all_objectives]

        # 最好点
        self.ideal_point = np.min(np.vstack((self.ideal_point, all_objectives)), axis=0)
        # 最差点
        self.nadir_point = np.max(np.vstack((self.nadir_point, all_objectives)), axis=0)

        # 归一化
        current_objectives = Normal.normalized(current_objectives, self.ideal_point, self.nadir_point)
        achieve_objectives = Normal.normalized(achieve_objectives, self.ideal_point, self.nadir_point)

        # 目标数据绑定参考向量。入口：参考向量（值和数量）、归一化的current目标值、current变量
        target, other_objectives_variables_pair = ASF.best_asf(reference_points=self.reference_points,
                                                               target_objectives=current_objectives,
                                                               target_variables=current_variables)

        # 当前代中非target部分添加到归档集中
        for objective, variable in other_objectives_variables_pair:
            achieve_objectives.append(objective)
            achieve_variables.append(variable)

        self.target = copy.deepcopy(target)
        # 在(interval)中除去target的剩余的解集
        self.associated = copy.deepcopy(achieve_variables)

        # 输出数据对
        self.datasetToNNetwork = ASF.ref_asf(reference_points=self.reference_points,
                                             associated_objectives=achieve_objectives,
                                             associated_variables=achieve_variables,
                                             target_variables=target)

    def get_with_compared_result_from_datasetToNNetwork(self):
        tmp = copy.deepcopy(self.datasetToNNetwork)
        with_compared_result_dataset = []
        for target, associated in tmp:
            if len(target) != 0 and len(associated) != 0:
                for every_item in associated:
                    with_compared_result_dataset.append([target, every_item])
        with_compared_result_dataset = self.non_compare.get_compare_result(with_compared_result_dataset)

        self.with_compare_result_datasetToNNetwork = with_compared_result_dataset

    def get_plotting_objectives_dataset(self):
        dataset1 = self.non_compare.get_dataset_for_plotting(copy.deepcopy(self.datasetToNNetwork))
        plotting_objectives_dataset = []
        for target, associate in dataset1:
            tmp1 = Normal.normalized([target], self.ideal_point, self.nadir_point)
            tmp1 = tmp1[0]
            tmp2 = []
            if len(associate) != 0:
                tmp2 = Normal.normalized(associate, self.ideal_point, self.nadir_point)
            plotting_objectives_dataset.append([tmp1, tmp2])

        self.plotting_objectives_dataset = plotting_objectives_dataset

    def get_observable_data(self) -> dict:
        """
            将数据通知给关联这个类的其他类使用
        Returns:

        """
        data = {
            "dataset": self             # 当前类实例
        }
        return data

    def register(self, observer):
        self.observable.register(observer)                  # 把输出对象加入主题中
