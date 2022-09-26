"""
读 MOKP数据，jMetalPy-master\resources\MOKP\MOKP_10_2_0.mokp---MOKP_10_3_0.mokp
"""

import random
# from typing import List
import numpy as np
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution

class MOKP(IntegerProblem):
    """ Class representing MO Knapsack Problem. """

    def __init__(self, filename: str = None):
        super(MOKP, self).__init__()

        self.__read_from_file(filename)

        self.lower_bound = [0] * self.number_of_variables
        self.upper_bound = [self.number_of_objectives] * self.number_of_variables
        self.obj_directions = [self.MAXIMIZE] * self.number_of_variables

    def __read_from_file(self, filename: str):
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
            lines = file.readlines()
            data = [line.split() for line in lines if len(line.split()) >= 1]

            self.number_of_variables = int(data[0][0])
            self.number_of_objectives = int(data[1][0])
            self.number_of_constraints = self.number_of_objectives

            # self.capacity = list(map(float, data[2]))  # 背包容量不同时，读入容量列表
            self.capacity = self.number_of_objectives * [float(data[2][0])]# 多个背包容量相同

            weights_and_profits = np.asfarray(data[self.number_of_objectives+1:], dtype=np.float32)
            self.profits = [weights_and_profits[i, :] for i in range(self.number_of_objectives)]
            self.weights = weights_and_profits[self.number_of_objectives, :]

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        total_profits = [0.0] * self.number_of_objectives
        total_weigths = [0.0] * self.number_of_objectives

        for i in range(self.number_of_variables):
            Knapsack_index = solution.variables[i]
            if Knapsack_index != 0:
                total_profits[Knapsack_index-1] += self.profits[Knapsack_index-1][i]
                total_weigths[Knapsack_index-1] += self.weights[i]

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
        new_solution = IntegerSolution(number_of_constraints = self.number_of_constraints,
                                      number_of_objectives = self.number_of_objectives,
                                       lower_bound = self.lower_bound,
                                       upper_bound = self.upper_bound)

        new_solution.variables = \
            [random.randint(self.lower_bound[i], self.upper_bound[i])
             for i in range(len(self.lower_bound))]

        return new_solution

    def get_name(self):
        return 'MOKP'
