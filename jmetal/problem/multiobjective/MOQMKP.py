
"""
数据来自 jMetalPy-master\resources\MO_QMKP\jeu_100_25_1.txt
"""
import copy
import os
import random

from AnalysisToolkit_LSH.MOQMKP_Package.createThreeSolutions import _create_three_solution
from jmetal.core.problem import IntegerProblem, PermutationProblem
from jmetal.core.solution import IntegerSolution, PermutationSolution

Max = 99999


class MOQMKP_PER(PermutationProblem):
    """ Class representing MOQMKP_PER. """

    create_counts = 0

    def __init__(self, filename: str = os.path.abspath('../../../0_MOQMKP/MOQMKP_data/TZB_10_2_100_1.txt')):
        super(MOQMKP_PER, self).__init__()

        self.number_of_items = 0      # 物品数
        self.number_of_Knapsacks = 0  # 背包数
        self.co_penalty = []          # 物物差异惩罚
        self.weights = []             # 物品重量
        self.capacity = []            # 包容量
        self.penalty = []             # 物包差异惩罚
        #LSH
        self.former_co_penalty = []
        self._create_three_solution = _create_three_solution
        #LSH
        self.filename = filename
        self.__read_from_file(self.filename)

        self.number_of_variables = self.number_of_items
        self.number_of_objectives = 3

        self.number_of_constraints = self.number_of_Knapsacks * 2    # 每个背包容量约束，每个背包充分装包约束

        self.constraints = [0.0] * self.number_of_constraints
        self.objectives = [0.0] * self.number_of_objectives
        # print(self.constraints)
        # print(self.objectives)

        # 三个目标：obj_0 重量最大化，obj_1 物包差异最小化，obj_2 物物差异最小化
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_directions[0] = self.MAXIMIZE

    def __read_from_file(self, filename: str = None):
        """
        This function reads a Knapsack Problem instance from a file.
        It expects the following format:

            num_of_items (dimension)
            capacity of the knapsack
            num_of_items-tuples of weight-profit

        :param filename: File which describes the instance.
        :type filename: str.
        """

        if filename is None:
            raise FileNotFoundError('Filename can not be None')

        with open(filename) as file:
            lines=file.readlines()

            data=[line.split() for line in lines if len(line.split())>=1]

            # 读Items数量
            self.number_of_items = int(data[1][0])
            self.number_of_Knapsacks = int(data[2][0])

            # 读Items装包费用
            self.penalty.clear()
            for line_index in range(self.number_of_Knapsacks):
                self.penalty.append(list(map(int, data[line_index + 3])))

            # LSH   读取初始Items间联合费用
            for line_index in range(self.number_of_items-1):
                tmp = list(map(int, data[line_index + 3 + self.number_of_Knapsacks]))
                self.former_co_penalty.append(tmp)

            # 读Items间联合费用
            for line_index in range(self.number_of_items-1):
                temp_list = []
                for col_index in range(line_index+1):
                    temp_list.append(-1)
                self.co_penalty.append(temp_list + list(map(int, data[line_index + 3 + self.number_of_Knapsacks])))

            # 联合费用最后一行
            temp_list = []
            for col_index in range(self.number_of_items):
                temp_list.append(-1)
            self.co_penalty.append(temp_list)

            for line_index in range(self.number_of_items):
                for col_index in range(line_index):
                    self.co_penalty[line_index][col_index] = (self.co_penalty[col_index][line_index])

            # 读包容量，非升序
            self.capacity = list(map(int, data[2+self.number_of_Knapsacks+self.number_of_items]))
            self.capacity.sort(reverse = True)
            # 读Items重量
            self.weights = list(map(int, data[3+self.number_of_Knapsacks+self.number_of_items]))

    def create_solution(self) -> PermutationSolution:
        """
        MOQMKP_PER.create_counts 为类属性（静态变量）比对象属性要好

        """
        MOQMKP_PER.create_counts += 1
        if MOQMKP_PER.create_counts == 1:
            new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                               number_of_objectives=self.number_of_objectives,
                                               number_of_constraints=self.number_of_constraints)
            new_solution.variables = self._create_three_solution.create_solution_item_weight_list(items_weight_list=self.weights)
            return new_solution
        elif MOQMKP_PER.create_counts == 2:
            new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                               number_of_objectives=self.number_of_objectives,
                                               number_of_constraints=self.number_of_constraints)
            new_solution.variables = self._create_three_solution.create_solution_item_item_penalty_list(
                items_weight=self.weights,
                knapsacks_weight=self.capacity,
                item_item_list=self.former_co_penalty
            )
            return new_solution
        elif MOQMKP_PER.create_counts == 3:
            new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                               number_of_objectives=self.number_of_objectives,
                                               number_of_constraints=self.number_of_constraints)
            new_solution.variables = self._create_three_solution.create_solution_item_knapsack_penalty_list(
                items_weight=self.weights,
                knapsacks_weight=self.capacity,
                item_knapsack_list=self.penalty
            )
            return new_solution
        else:
            new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                               number_of_objectives=self.number_of_objectives,
                                               number_of_constraints=self.number_of_constraints)

            new_solution.variables = random.sample(range(self.number_of_variables), k=self.number_of_variables)
            return new_solution

    # 装包启发式
    def permut_to_int (self, solution: PermutationSolution):
        solution.variables_int = [-1] * self.number_of_variables   # 索引对应物品，表示物品是否装包，（-1）-未装，int-表示装入哪个包
        knapsacks_total_weights = [0.0] * self.number_of_Knapsacks  #动态记录包总重量
        for knap_i in range(self.number_of_Knapsacks):
           for var_i in range(self.number_of_variables):
               item_i = solution.variables[var_i]
               # 适配/包余量够/物品未装包
               if self.penalty[knap_i][item_i] < Max \
                   and knapsacks_total_weights[knap_i] + self.weights[item_i] <= self.capacity[knap_i] \
                       and solution.variables_int[item_i] == -1 :
                   solution.variables_int[item_i] = knap_i   # 物品装当前背包
                   knapsacks_total_weights[knap_i] += self.weights[item_i] # 更新背包总重量

        return solution

    def evaluate(self, solution:PermutationSolution) -> PermutationSolution:
        """permutation to integer"""
        solution = self.permut_to_int(solution)

        """evaluate the solution"""
        total_penalty = [0.0] * self.number_of_Knapsacks
        total_weights = [0.0] * self.number_of_Knapsacks
        total_co_penalty = [0.0] * self.number_of_Knapsacks

        In_knapsack = [[] for i in range(self.number_of_Knapsacks)]  # 记录装入包中的物品编号
        for item_i in range(self.number_of_variables):
            Knap_index = solution.variables_int[item_i]
            if Knap_index != -1:
                total_penalty[Knap_index] += self.penalty[Knap_index][item_i]
                total_weights[Knap_index] += self.weights[item_i]
                In_knapsack[Knap_index].append(item_i)
        # print(In_knapsack)
        # 计算Items间联合费用

        for knap_i in range(self.number_of_Knapsacks):
            for index_i in range(len(In_knapsack[knap_i])-1):
                for index_j in range(index_i + 1, len(In_knapsack[knap_i])):
                    total_co_penalty[knap_i] += self.co_penalty[In_knapsack[knap_i][index_i]][In_knapsack[knap_i][index_j]]
        # print(total_co_penalty)
        # print("-------")

        # 初始化
        solution.constraints = [0.0] * self.number_of_constraints
        solution.objectives = [0.0] * self.number_of_objectives

        for knap_i in range(self.number_of_Knapsacks):
            solution.constraints[knap_i] = self.capacity[knap_i] - total_weights[knap_i]  # 背包容量约束，看背包剩余容量，此处>=0，则满足约束
            # 背包剩余容量需小于未装包且适配的最小物品重量
            solution.constraints[self.number_of_Knapsacks + knap_i] = 1  #初始化
            for item_i in range(self.number_of_items):
                if self.penalty[knap_i][item_i] < Max and solution.variables_int[item_i] == -1 \
                    and self.weights[item_i] <= solution.constraints[knap_i] :
                    solution.constraints[self.number_of_Knapsacks + knap_i] = -1 # 1-满足约束，-1-不满足约束

        for knap_i in range(self.number_of_Knapsacks):
            solution.objectives[0] += total_weights[knap_i] * self.obj_directions[0]  # 装包总重量
            solution.objectives[1] += total_penalty[knap_i] * self.obj_directions[1]  # 装包总差异惩罚
            solution.objectives[2] += total_co_penalty[knap_i] * self.obj_directions[2]  # 物品联合惩罚

        for cons_i in range(self.number_of_constraints):
            if solution.constraints[cons_i] < 0:  # 违反约束处理
                solution.objectives[0] = 0
                solution.objectives[1] = Max
                solution.objectives[2] = Max
                break

        return solution

    def get_name(self):
        return 'MOQMKP_PER'

    def get_number_variables(self):
        return self.number_of_variables



class MOQMKP_INT(IntegerProblem):
    """ Class representing MOQMKP_INT. """
    # 加完成约束后还没改

    def __init__(self, filename: str = os.path.abspath('../../../0_MOQMKP/MOQMKP_data/TZB_10_2_100_1.txt')):
        super(MOQMKP_INT, self).__init__()

        self.number_of_items = 0    # 物品数
        self.number_of_Knapsacks = 0  # 背包数
        self.weights = []   # 物品重量
        self.capacity = []  # 包容量
        self.penalty = []   # 物品背包差异最小化
        self.co_penalty = []  # 物品间差异惩罚

        self.filename = filename
        self.__read_from_file(self.filename)

        self.number_of_variables = self.number_of_items
        self.number_of_objectives = 3
        self.lower_bound=[0]*self.number_of_variables
        self.upper_bound=[self.number_of_Knapsacks]*self.number_of_variables
        self.number_of_constraints = self.number_of_Knapsacks

        # 三个目标：obj_0 重量最大化，obj_1 物品背包差异最小化，obj_2 物品间联合差异惩罚最小化
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_directions[0] = self.MAXIMIZE

        # self.kmin = []
        # self.kmax = []
        # # kmax[0,self.number_of_Knapsacks-1]/kmin 存储每个背包最多/少能装多少个物品
        # # kmax[self.number_of_Knapsacks]/kmin 存储背包容量总和最多/少能装多少个物品
        # self.kmin, self.kmax = self.get_k_bound()

    def __read_from_file(self, filename: str = None):
        """
        This function reads a Knapsack Problem instance from a file.
        It expects the following format:

            num_of_items (dimension)
            capacity of the knapsack
            num_of_items-tuples of weight-profit

        :param filename: File which describes the instance.
        :type filename: str.
        """

        if filename is None:
            raise FileNotFoundError('Filename can not be None')

        with open(filename) as file:
            lines=file.readlines()
            data=[line.split() for line in lines if len(line.split())>=1]

            # 读Items数量
            self.number_of_items = int(data[1][0])
            self.number_of_Knapsacks = int(data[2][0])

            # 读Items装包费用
            self.penalty.clear()
            for line_index in range(self.number_of_Knapsacks):
                self.penalty.append(list(map(int, data[line_index + 3])))

            # 读Items间联合费用
            for line_index in range(self.number_of_items-1):
                temp_list = []
                for col_index in range(line_index+1):
                    temp_list.append(-1)
                self.co_penalty.append(temp_list + list(map(int, data[line_index + 3 + self.number_of_Knapsacks])))
            # 联合费用最后一行
            temp_list = []
            for col_index in range(self.number_of_items):
                temp_list.append(-1)
            self.co_penalty.append(temp_list)

            for line_index in range(self.number_of_items):
                for col_index in range(line_index):
                    self.co_penalty[line_index][col_index] = (self.co_penalty[col_index][line_index])

            # 读包容量
            self.capacity = list(map(int, data[2+self.number_of_Knapsacks+self.number_of_items]))
            # 读Items重量
            self.weights = list(map(int, data[3+self.number_of_Knapsacks+self.number_of_items]))

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        total_penalty = [0.0] * self.number_of_constraints
        total_weigths = [0.0] * self.number_of_constraints
        total_co_penalty = [0.0] * self.number_of_constraints

        In_knapsack = [[] for i in range(self.number_of_constraints)] # 记录装入包中的物品编号
        for i in range(self.number_of_variables):
            Knapsack_index = solution.variables[i]
            if Knapsack_index != 0:
                total_penalty[Knapsack_index - 1] += self.penalty[Knapsack_index - 1][i]
                total_weigths[Knapsack_index-1] += self.weights[i]
                In_knapsack[Knapsack_index-1].append(i)

        # 计算Items间联合费用
        for i in range(self.number_of_constraints):
            for j in range(len(In_knapsack[i])-1):
                for k in range(j+1, len(In_knapsack[i])):
                    total_co_penalty[i] += self.co_penalty[In_knapsack[i][k]][In_knapsack[i][j]]

        for i in range(self.number_of_Knapsacks):
            solution.constraints[i] = self.capacity[i] - total_weigths[i]

        for i in range(self.number_of_objectives):
            solution.objectives[i] = 0
        for i in range(self.number_of_constraints):
            if solution.constraints[i]<0:  # 违反约束处理
                solution.objectives[0] = 0
                solution.objectives[1] = Max
                solution.objectives[2] = Max
                break
            else:
                solution.objectives[0] += total_weigths[i]*self.obj_directions[0] # 装包总重量
                solution.objectives[1] += total_penalty[i] * self.obj_directions[1] # 装包总差异惩罚
                solution.objectives[2] += total_co_penalty[i] * self.obj_directions[2]  # 物品联合惩罚

        return solution

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(number_of_constraints = self.number_of_constraints,
                                      number_of_objectives = self.number_of_objectives,
                                       lower_bound = self.lower_bound,
                                       upper_bound = self.upper_bound)

        new_solution.variables = \
            [random.randint(self.lower_bound[i], self.upper_bound[i]) for i in range(self.number_of_variables)]

        return new_solution

    # 获取装包物品的个数上下界
    # 总体最多能装下的个数上下界：kmin/kmax[-1]包重求和装入（不准确）
    def get_k_bound(self):
        weights = copy.deepcopy(self.weights)
        weights.sort()  # 物品重量升序

        # 多包总容量
        all_capacity = 0
        for i in range(len(self.capacity)):
            all_capacity += self.capacity[i]

        # 测试用，统计物品总重量
        # sum_weights = 0
        # for i in range(self.number_of_variables):
        #     sum_weights += weights[i]

        # 计算最多装入多少个
        # kmax[0,self.number_of_Knapsacks-1]存储每个背包最多能装多少个物品
        # kmax[self.number_of_Knapsacks]存储背包容量总和最多能装多少个物品
        kmax = []
        for knapsack_i in range(self.number_of_Knapsacks):
            sum_weights = 0
            for item_i in range(self.number_of_items):
                sum_weights += weights[item_i]
                if(sum_weights == self.capacity[knapsack_i]):
                    kmax.append(item_i+1)
                    break
                elif(sum_weights > self.capacity[knapsack_i]):
                    kmax.append(item_i)
                    break
                else:
                    continue

        sum_weights = 0
        for item_i in range(self.number_of_items):
            sum_weights += weights[item_i]
            if (sum_weights == all_capacity):
                kmax.append(item_i + 1)
                break
            elif (sum_weights > all_capacity):
                kmax.append(item_i)
                break
            else:
                continue

        # 计算最少装入多少个
        # kmin[0,self.number_of_Knapsacks-1]存储每个背包最少能装多少个物品
        # kmin[self.number_of_Knapsacks]存储背包容量总和最少能装多少个物品
        kmin = []
        for knapsack_i in range(self.number_of_Knapsacks):
            sum_weights = 0
            for item_i in range(self.number_of_items):
                sum_weights += weights[self.number_of_items - 1 - item_i]
                if (sum_weights == self.capacity[knapsack_i]):
                    kmin.append(item_i+1)
                    break
                elif (sum_weights > self.capacity[knapsack_i]):
                    kmin.append(item_i)
                    break
                else:
                    continue
        sum_weights = 0
        for item_i in range(self.number_of_items):
            sum_weights += weights[self.number_of_items - 1 - item_i]
            if (sum_weights == all_capacity):
                kmin.append(item_i + 1)
                break
            elif (sum_weights > all_capacity):
                kmin.append(item_i)
                break
            else:
                continue

        return kmin, kmax

    def get_name(self):
        return 'MOQMKP_INT'





