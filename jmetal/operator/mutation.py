import copy
import random

from jmetal.core.operator import Mutation
from jmetal.core.solution import BinarySolution, Solution, FloatSolution, IntegerSolution, PermutationSolution, \
    CompositeSolution
from jmetal.util.ckecking import Check

"""
.. module:: mutation
   :platform: Unix, Windows
   :synopsis: Module implementing mutation operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NullMutation(Mutation[Solution]):

    def __init__(self):
        super(NullMutation, self).__init__(probability=0)

    def execute(self, solution: Solution) -> Solution:
        return solution

    def get_name(self):
        return 'Null mutation'


class BitFlipMutation(Mutation[BinarySolution]):

    def __init__(self, probability: float):
        super(BitFlipMutation, self).__init__(probability=probability)

    def execute(self, solution: BinarySolution) -> BinarySolution:
        Check.that(type(solution) is BinarySolution, "Solution type invalid")

        for i in range(solution.number_of_variables):
            for j in range(len(solution.variables[i])):
                rand = random.random()
                if rand <= self.probability:
                    solution.variables[i][j] = True if solution.variables[i][j] is False else False

        return solution

    def get_name(self):
        return 'BitFlip mutation'


class PolynomialMutation(Mutation[FloatSolution]):

    def __init__(self, probability: float, distribution_index: float = 0.20):
        super(PolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")
        for i in range(solution.number_of_variables):
            rand = random.random()

            if rand <= self.probability:
                y = solution.variables[i]
                yl, yu = solution.lower_bound[i], solution.upper_bound[i]

                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    rnd = random.random()
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, self.distribution_index + 1.0))
                        deltaq = pow(val, mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, self.distribution_index + 1.0))
                        deltaq = 1.0 - pow(val, mut_pow)

                    y += deltaq * (yu - yl)
                    if y < solution.lower_bound[i]:
                        y = solution.lower_bound[i]
                    if y > solution.upper_bound[i]:
                        y = solution.upper_bound[i]

                solution.variables[i] = y

        return solution

    def get_name(self):
        return 'Polynomial mutation'


class IntegerPolynomialMutation(Mutation[IntegerSolution]):

    def __init__(self, probability: float, distribution_index: float = 0.20):
        super(IntegerPolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        Check.that(issubclass(type(solution), IntegerSolution), "Solution type invalid")

        for i in range(solution.number_of_variables):
            if random.random() <= self.probability:
                y = solution.variables[i]
                yl, yu = solution.lower_bound[i], solution.upper_bound[i]

                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    rnd = random.random()
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.distribution_index + 1.0))
                        deltaq = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (self.distribution_index + 1.0))
                        deltaq = 1.0 - val ** mut_pow

                    y += deltaq * (yu - yl)
                    if y < solution.lower_bound[i]:
                        y = solution.lower_bound[i]
                    if y > solution.upper_bound[i]:
                        y = solution.upper_bound[i]

                solution.variables[i] = int(round(y))
        return solution

    def get_name(self):
        return 'Polynomial mutation (Integer)'


class SimpleRandomMutation(Mutation[FloatSolution]):

    def __init__(self, probability: float):
        super(SimpleRandomMutation, self).__init__(probability=probability)

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(type(solution) is FloatSolution, "Solution type invalid")

        for i in range(solution.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                solution.variables[i] = solution.lower_bound[i] + \
                                        (solution.upper_bound[i] - solution.lower_bound[i]) * random.random()
        return solution

    def get_name(self):
        return 'Simple random_search mutation'


class UniformMutation(Mutation[FloatSolution]):

    def __init__(self, probability: float, perturbation: float = 0.5):
        super(UniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(type(solution) is FloatSolution, "Solution type invalid")

        for i in range(solution.number_of_variables):
            rand = random.random()

            if rand <= self.probability:
                tmp = (random.random() - 0.5) * self.perturbation
                tmp += solution.variables[i]

                if tmp < solution.lower_bound[i]:
                    tmp = solution.lower_bound[i]
                elif tmp > solution.upper_bound[i]:
                    tmp = solution.upper_bound[i]

                solution.variables[i] = tmp

        return solution

    def get_name(self):
        return 'Uniform mutation'


class NonUniformMutation(Mutation[FloatSolution]):

    def __init__(self, probability: float, perturbation: float = 0.5, max_iterations: int = 0.5):
        super(NonUniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(type(solution) is FloatSolution, "Solution type invalid")

        for i in range(solution.number_of_variables):
            if random.random() <= self.probability:
                rand = random.random()

                if rand <= 0.5:
                    tmp = self.__delta(solution.upper_bound[i] - solution.variables[i], self.perturbation)
                else:
                    tmp = self.__delta(solution.lower_bound[i] - solution.variables[i], self.perturbation)

                tmp += solution.variables[i]

                if tmp < solution.lower_bound[i]:
                    tmp = solution.lower_bound[i]
                elif tmp > solution.upper_bound[i]:
                    tmp = solution.upper_bound[i]

                solution.variables[i] = tmp

        return solution

    def set_current_iteration(self, current_iteration: int):
        self.current_iteration = current_iteration

    def __delta(self, y: float, b_mutation_parameter: float):
        return (y * (1.0 - pow(random.random(),
                               pow((1.0 - 1.0 * self.current_iteration / self.max_iterations), b_mutation_parameter))))

    def get_name(self):
        return 'Uniform mutation'



class PermutationSwapMutation(Mutation[PermutationSolution]):

    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        Check.that(type(solution) is PermutationSolution, "Solution type invalid")

        rand = random.random()

        if rand <= self.probability:
            pos_one, pos_two = random.sample(range(solution.number_of_variables - 1), 2)
            solution.variables[pos_one], solution.variables[pos_two] = \
                solution.variables[pos_two], solution.variables[pos_one]

        return solution

    def get_name(self):
        return 'Permutation Swap mutation'

# 组合多种算子
class CompositeMutation(Mutation[Solution]):
    def __init__(self, mutation_operator_list: [Mutation]):
        super(CompositeMutation, self).__init__(probability=1.0)

        Check.is_not_none(mutation_operator_list)
        Check.collection_is_not_empty(mutation_operator_list)

        self.mutation_operators_list = []
        for operator in mutation_operator_list:
            Check.that(issubclass(operator.__class__, Mutation), "Object is not a subclass of Mutation")
            self.mutation_operators_list.append(operator)

    def execute(self, solution: CompositeSolution) -> CompositeSolution:
        Check.is_not_none(solution)

        mutated_solution_components = []
        for i in range(solution.number_of_variables):
            mutated_solution_components.append(self.mutation_operators_list[i].execute(solution.variables[i]))

        return CompositeSolution(mutated_solution_components)

    def get_name(self) -> str:
        return "Composite mutation operator"


class ScrambleMutation(Mutation[PermutationSolution]):

    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        for i in range(solution.number_of_variables):
            rand = random.random()

            if rand <= self.probability:
                point1 = random.randint(0, len(solution.variables[i]))
                point2 = random.randint(0, len(solution.variables[i]) - 1)

                # 使得point2 > point1
                if point2 >= point1:
                    point2 += 1
                else:
                    point1, point2 = point2, point1
                # 基因串不能超过20，为啥？
                if point2 - point1 >= 20:
                    point2 = point1 + 20

                values = solution.variables[i][point1:point2]
                solution.variables[i][point1:point2] = random.sample(values, len(values))

        return solution

    def get_name(self):
        return 'Scramble'

# TZB 2022.1.1 离散变异，方法来自安腾腾
class PDDEMutation(Mutation[PermutationSolution]):

    def __init__(self, CR: float):
        super(PDDEMutation, self).__init__(probability = 1.0)
        self.CR = CR
        self.current_individual: PermutationSolution = None

    def Custom_Subtract(self, SubList: [PermutationSolution]) -> [PermutationSolution]:
        res = copy.deepcopy(SubList[1])
        for i in range(len(SubList[1])):
            res[i] = self.Indexof(SubList[1], SubList[0][i])
        return res

    def Custom_Add(self, AddList: [PermutationSolution]) -> [PermutationSolution]:
        res = copy.deepcopy(AddList[0])
        for i in range(len(AddList[0])):
            res[i] = AddList[0][AddList[1][i]]
        return  res

    def Indexof(self,Per: [PermutationSolution], Index: int) -> int:
        for i in range(len(Per)):
            if(Per[i] == Index):
                return i
        return -2

    def execute(self, parents: [PermutationSolution]) -> [PermutationSolution]:
        """ Execute the differential evolution crossover ('best/1/bin' variant in jMetal).
        """
        if len(parents) != self.get_number_of_parents():
            raise Exception('The number of parents is not {}: {}'.format(self.get_number_of_parents(), len(parents)))

        number_of_variables = parents[0].number_of_variables

        try_ind = copy.deepcopy(parents[0])

        if random.random() < self.CR:
            sub_list = []
            sub_list.append(parents[1].variables)
            sub_list.append(parents[2].variables)
            res_sub = self.Custom_Subtract(sub_list)
            add_list = []
            add_list.append(parents[0].variables)
            add_list.append(res_sub)
            res_add = self.Custom_Add(add_list)
            try_ind.variables = res_add
            try_ind.objectives.clear()
            try_ind.constraints.clear()

        return [try_ind]

    def get_number_of_parents(self) -> int:
        return 3

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self):
        return 'PDDE mutation operator'

