# 读MOEA/D MKP数据
# jMetalPy-master\resources\MOKP\knapsack_100.2.txt--knapsack_750.4.txt

import os
import random
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution

class MOKP(IntegerProblem):
    """ Class representing MO Knapsack Problem. """

    def __init__(self, filename: str = os.path.abspath('..\\..\\resources\\MOKP\\knapsack_100.2.txt')):
        super(MOKP, self).__init__()

        self.__read_from_file(filename)

        self.number_of_objectives = len(self.capacity)
        self.number_of_constraints = self.number_of_objectives
        self.lower_bound = [0] * self.number_of_variables
        self.upper_bound = [self.number_of_objectives] * self.number_of_variables
        self.obj_directions = [self.MAXIMIZE] * self.number_of_variables

    def __read_from_file(self, filename: str = os.path.abspath('..\\..\\resources\\MOKP\\knapsack_100.2.txt')):
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
        b = 0
        weight_list = []
        weight_list1 = []
        profit_list = []
        profit_list1 = []
        capacity_list = []
        with open(filename) as file:
            lines = file.readlines()
            data = [line.split() for line in lines if len(line.split()) >= 1]
            for data1 in data:
                if "specification" in data1:
                    number_of_variables = int(data1[5])
                if "weight" in data1[0]:
                    weight_list.append(float(data1[1][1:]))
                    b = b + 1
                    if b % number_of_variables == 0:
                        weight_list1.append(weight_list)
                        weight_list = []
                        b = 0

                if "capacity" in data1[0]:
                    capacity_list.append(float(data1[1][1:]))

            for data1 in data:
                if "profit" in data1[0]:
                    profit_list.append(float(data1[1][1:]))
                    b = b + 1
                    if b % number_of_variables == 0:
                        b = 0
                        profit_list1.append(profit_list)
                        profit_list = []

        self.number_of_variables = number_of_variables
        self.capacity = capacity_list
        self.weights = weight_list1
        self.profits = profit_list1

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        total_profits = [0.0] * self.number_of_objectives
        total_weigths = [0.0] * self.number_of_objectives

        for i in range(self.number_of_variables):
            Knapsack_index = solution.variables[i]
            if Knapsack_index != 0:
                total_profits[Knapsack_index - 1] += self.profits[Knapsack_index - 1][i]
                total_weigths[Knapsack_index - 1] += self.weights[Knapsack_index - 1][i]

        for i in range(self.number_of_constraints):
            solution.constraints[i] = self.capacity[i] - total_weigths[i]  # 违反约束，总价值为0

        for i in range(self.number_of_constraints):
            if solution.constraints[i] < 0:
                for j in range(self.number_of_constraints):  # 所有包价值置-1
                    total_profits[j] = -1
                break

        for i in range(self.number_of_objectives):
            solution.objectives[i] = self.obj_directions[i] * total_profits[i]

        return solution

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(number_of_constraints=self.number_of_constraints,
                                       number_of_objectives=self.number_of_objectives,
                                       lower_bound=self.lower_bound,
                                       upper_bound=self.upper_bound)

        new_solution.variables = \
            [random.randint(self.lower_bound[i], self.upper_bound[i])
             for i in range(len(self.lower_bound))]

        return new_solution

    def get_name(self):
        return 'MOKP'
