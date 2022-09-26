import copy
import os
import random
import math
from enum import Enum
import numpy as np
from typing import TypeVar, Generic, List
from abc import ABC, abstractmethod

# logger = logging.getLogger("AdditionalTools")


class DensityEnum(Enum):
    Density_1_00 = 1.0
    Density_0_75 = 0.75
    Density_0_50 = 0.50
    Density_0_25 = 0.25


class FileCountEnum(Enum):
    File_Counts_1 = 1
    File_Counts_25 = 25
    File_Counts_50 = 50


class ValueEnum(Enum):
    Max = 99999
    Min = -99999


S = TypeVar("S")
R = TypeVar("R")


class Generator(Generic[S, R], ABC):
    def __init__(self,
                 number_of_variables: int,
                 number_of_knapsacks: int,
                 density: DensityEnum,
                 max_file_counts: FileCountEnum,
                 item_wei_lb: S,
                 item_wei_ub: S
                 ):
        self.number_of_variables = number_of_variables      # 每个背包固定的物品个数
        self.number_of_knapsacks = number_of_knapsacks      # 背包个数
        self.density = density.value                        # 合法值占空比
        self.max_file_counts = max_file_counts.value        # 最大循环次数
        self.item_wei_lb = item_wei_lb                      # 物品重量 下限  lower_bound
        self.item_wei_ub = item_wei_ub                      # 物品重量 上限  upper_bound
        self.accumulate_counts = 0                          # 累计次数

        # self.logging_file_formatter = str("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 程序运行方法
    def run(self):
        while not self.stopping_condition_is_met(self.accumulate_counts):
            self.step()
            self.accumulate_counts += 1

    # 终止条件
    def stopping_condition_is_met(self, accumulate_counts: int) -> bool:
        if accumulate_counts >= self.max_file_counts:
            return True
        return False

    # 创建文件夹
    def generate_folder(self, folder: str):
        is_existed_folder = os.path.exists(folder)
        if not is_existed_folder:
            try:
                os.makedirs(folder)
            except OSError:
                raise Exception(F"{folder}文件夹创建失败")

    @abstractmethod
    def step(self):
        """
            核心方法，将数据生成的过程用step函数表示
        Returns:        None

        """
        pass

    @abstractmethod
    def generate_item_weights_list(self) -> List:
        """
            生成物品重量的链表数据
        Returns:    List

        """
        pass

    @abstractmethod
    def generate_file(self, filename: str):
        """ 产生符合要求的数据文件 """
        pass

    @abstractmethod
    def get_name(self):
        pass


class MOQMKPGenerator(Generator[int, int], ABC):
    def __init__(self,
                 number_of_variables: int,
                 number_of_knapsacks: int,
                 density: DensityEnum,
                 max_file_counts: FileCountEnum,
                 cost_lb: int,
                 cost_ub: int,
                 union_cost_lb: int,
                 union_cost_ub: int,
                 item_wei_lb: int,
                 item_wei_ub: int
                 ):
        super(MOQMKPGenerator, self).__init__(number_of_variables=number_of_variables,
                                              number_of_knapsacks=number_of_knapsacks,
                                              density=density,
                                              max_file_counts=max_file_counts,
                                              item_wei_lb=item_wei_lb,
                                              item_wei_ub=item_wei_ub)

        self.cost_lb = cost_lb                          # 物品背包差异惩罚 下限   lower_bound
        self.cost_ub = cost_ub                          # 物品背包差异惩罚 上限   upper_bound
        self.union_cost_lb = union_cost_lb              # 物品与物品差异惩罚 下限
        self.union_cost_ub = union_cost_ub

        self.item_weight_list = []                      # 物品重量链表
        self.knapsacks_weight_list = []                 # 背包重量链表
        self.item_knapsack_penalty = []                 # 物品与背包差异惩罚
        self.item_item_penalty = []                     # 物品与物品差异惩罚
        self.max_value = ValueEnum.Max.value            # 99999

    def step(self):
        self.item_weight_list = []
        self.knapsacks_weight_list = []
        self.item_knapsack_penalty = []
        self.item_item_penalty = []

        self.generate_item_weights_list()
        self.generate_knapsack_weights_list(self.item_weight_list)
        self.generate_item_item_penalty()
        self.generate_item_knapsack_penalty()
        self.generate_file()

    def generate_file(self):
        density_value = int(self.density * 100)
        folder1 = f"{self.get_name()}_Data"
        folder2 = f"{folder1}/Density_{density_value}"
        folder3 = f"{folder2}/{self.number_of_variables}_{self.number_of_knapsacks}"
        self.generate_folder(folder1)
        self.generate_folder(folder2)
        self.generate_folder(folder3)

        filename = "{0}/TZB_{1}_{2}_{3}_{4}".format(
            folder3, self.number_of_variables, self.number_of_knapsacks,
            density_value, self.accumulate_counts + 1
        )

        with open(filename, 'w+', encoding='utf-8') as f:
            f.write(filename + '\n')                                # 路径

            f.write(str(self.number_of_variables) + '\n')           # 物品数
            f.write(str(self.number_of_knapsacks) + '\n')           # 背包数

            for for_i in range(self.number_of_knapsacks):           # 物品与背包差异惩罚
                counts = len(self.item_knapsack_penalty[for_i])
                for for_j in range(counts):
                    if for_j == counts - 1:
                        f.write(str(self.item_knapsack_penalty[for_i][for_j]) + '\n')
                    else:
                        f.write(str(self.item_knapsack_penalty[for_i][for_j]) + ' ')
            f.write('\n')

            for for_i in range(self.number_of_variables - 1):                 # 物品与物品差异惩罚
                counts = len(self.item_item_penalty[for_i])
                for for_j in range(counts):
                    if for_j == counts - 1:
                        f.write(str(self.item_item_penalty[for_i][for_j]) + '\n')
                    else:
                        f.write(str(self.item_item_penalty[for_i][for_j]) + ' ')
            f.write('\n')

            for for_i in range(self.number_of_knapsacks):          # 背包重量
                if for_i == self.number_of_knapsacks - 1:
                    f.write(str(self.knapsacks_weight_list[for_i]) + '\n')
                else:
                    f.write(str(self.knapsacks_weight_list[for_i]) + ' ')

            for for_i in range(self.number_of_variables):           # 物品重量
                if for_i == self.number_of_variables - 1:
                    f.write(str(self.item_weight_list[for_i]) + '\n')
                else:
                    f.write(str(self.item_weight_list[for_i]) + ' ')

            f.write('\n')
            f.write(str("Density:" + str(int(self.density * 100)) + '%'))

    def generate_item_weights_list(self):
        self.item_item_penalty = []
        self.item_weight_list = [random.randint(self.item_wei_lb, self.item_wei_ub)
                                 for _ in range(self.number_of_variables)]

    def generate_knapsack_weights_list(self, item_weight_list: List):
        self.knapsacks_weight_list = []
        all_item_weights = 0
        for item_weight_value in item_weight_list:
            all_item_weights += item_weight_value
        all_item_weights = int(all_item_weights / self.number_of_knapsacks)
        self.knapsacks_weight_list = [random.randint(100, all_item_weights) for _ in range(self.number_of_variables)]

    def judge_whether_item_knapsack_penalty_is_legal(self, item_knapsack_list: List) -> bool:
        _item_knapsack_penalty = copy.deepcopy(item_knapsack_list)
        _item_knapsack_penalty = np.mat(_item_knapsack_penalty)
        _item_knapsack_penalty = np.transpose(_item_knapsack_penalty).tolist()
        # print(_item_knapsack_penalty)

        for item in _item_knapsack_penalty:
            is_only_had_max_value = set(item)
            if len(is_only_had_max_value) == 1 and is_only_had_max_value.pop() == self.max_value:            # 只有 99999
                return False
        return True

    # def get_index_where_column_is_invalid(self, item_knapsack_list: List) -> int:
    #     _item_knapsack_penalty = copy.deepcopy(item_knapsack_list)
    #     _item_knapsack_penalty = np.mat(_item_knapsack_penalty)
    #     _item_knapsack_penalty = np.transpose(_item_knapsack_penalty).tolist()
    #     for index in range(len(_item_knapsack_penalty)):
    #         is_only_had_max_value = set(_item_knapsack_penalty[index])
    #         if len(is_only_had_max_value) == 1 and is_only_had_max_value.pop() == self.max_value:  # 只有 99999
    #             print("index" + str(index))
    #             return index
    #     return -1

    def generate_item_knapsack_penalty(self):
        self.item_knapsack_penalty = []
        if self.density == DensityEnum.Density_1_00.value:
            for for_i in range(self.number_of_knapsacks):
                single_penalty = [random.randint(self.cost_lb, self.cost_ub) for _ in range(self.number_of_variables)]
                self.item_knapsack_penalty.append(single_penalty)
        else:
            # Density: 0.75,0.50,0.25
            all_counts = self.number_of_knapsacks * self.number_of_variables                # 总个数
            valid_value_counts = math.floor(all_counts * self.density)                      # 合法值个数
            invalid_value_counts = all_counts - valid_value_counts                          # 非法值个数
            for for_i in range(self.number_of_knapsacks):                                   # 初始化全部为 max_value = 99999
                self.item_knapsack_penalty.append([self.max_value for _ in range(self.number_of_variables)])

            # 情况一: 合法值小于个数一行的元素，得出的结果是，每一列至少有一个合法值元素
            if valid_value_counts <= self.number_of_variables:
                for for_j in range(self.number_of_variables):
                    random_valid_value = random.randint(self.cost_lb, self.cost_ub)  # 随机值
                    self.item_knapsack_penalty[random.randint(0, self.number_of_knapsacks-1)][for_j] = random_valid_value
            else:
                """"
                情况二：合法值个数大于一行的元素
                利用字典的方法，随即在字典中抽出i,j下标进行运算。
                """
                # 初始化字典
                init_dictionart = {x: [_ for _ in range(self.number_of_variables)] for x in range(self.number_of_knapsacks)}
                # print(init_dictionart)

                # 确保每一列只要有一个合法值
                for for_i in range(self.number_of_variables):               # 遍历每一列
                    random_row_index = random.randint(0, self.number_of_knapsacks - 1)          # 随机生成行下标
                    _formal_row_list = list(init_dictionart[random_row_index])                  # 将字典中的list赋值到_formal_row_list
                    random_valid_value = random.randint(self.cost_lb, self.cost_ub)             # 合法值随机值
                    self.item_knapsack_penalty[random_row_index][for_i] = random_valid_value    # 原先都是max_value，直接替换
                    valid_value_counts -= 1                                                     # 合法值个数减一
                    _formal_row_list.remove(for_i)             # remove 替换的下标，[random_row_index][for_i] = 合法值，无需在字典中存在
                    init_dictionart[random_row_index] = _formal_row_list            # 替换list

                """
                    我们在init_dictionart[random_row_index]的values中抽出非合法值（即self.max_value = 99999)的下标，大大提高了随机的效率
                    降低了随机出无效的的下标
                """
                while valid_value_counts > 0:
                    random_row_index = random.randint(0, self.number_of_knapsacks - 1)
                    random_column_index_list = random.sample(init_dictionart[random_row_index], k=random.randint(0, len(init_dictionart[random_row_index]) - 1))
                    _formal_row_list = list(init_dictionart[random_row_index])
                    for index in random_column_index_list:
                        if self.item_knapsack_penalty[random_row_index][index] == self.max_value:
                            random_valid_value = random.randint(self.cost_lb, self.cost_ub)  # 随机值
                            self.item_knapsack_penalty[random_row_index][index] = random_valid_value
                            _formal_row_list.remove(index)
                            valid_value_counts -= 1
                    init_dictionart[random_row_index] = _formal_row_list

        if self.judge_whether_item_knapsack_penalty_is_legal(self.item_knapsack_penalty) is False:
            print(F"第{self.accumulate_counts}次，生成的item_knapsack_penalty[](物品与背包差异惩罚）错误")

    def get_name(self):
        return "MOQMKP_Generator"

    def generate_item_item_penalty(self):               # 三角矩阵
        self.item_item_penalty = []
        for x in range(self.number_of_variables - 1):
            row_penalty = [random.randint(self.union_cost_lb, self.union_cost_ub) for _ in range(self.number_of_variables - x - 1)]
            self.item_item_penalty.append(row_penalty)

    # def generate_item_item_penalty(self):               # 对称矩阵
    #     self.item_item_penalty = []
    #     for x in range(self.number_of_variables):
    #         row_penalty = [random.randint(self.union_cost_lb, self.union_cost_ub) for _ in range(self.number_of_variables)]
    #         self.item_item_penalty.append(row_penalty)
    #
    #     _tmp1 = copy.deepcopy(self.item_item_penalty)
    #     _tmp1 = np.mat(_tmp1)
    #     _tmp1 = np.triu(_tmp1, 1)           # 上三角矩阵
    #     _tmp2 = copy.deepcopy(_tmp1)
    #     _tmp2 = _tmp2.transpose()
    #
    #     self.item_item_penalty = _tmp1 + _tmp2
    #     np.fill_diagonal(self.item_item_penalty, -1)
