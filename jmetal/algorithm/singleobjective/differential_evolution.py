
"""
TZB 2021.10
"""
import random
import copy
from typing import TypeVar, List

from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.operator import Crossover, Selection, Mutation
from jmetal.core.problem import Problem
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

class DifferentialEvolution(EvolutionaryAlgorithm[S, R]):
    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator
                 ):
        super(DifferentialEvolution, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size)

        self.selection_operator = selection
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.dominance_comparator = store.default_comparator # 比较两个解


        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        # 根据交配过程对父代和子代得数量需求，确定mating_pool。一般2个父代产生2个子代
        # self.mating_pool_size = \
        #     self.offspring_population_size * \
        #     self.crossover_operator.get_number_of_parents() // self.crossover_operator.get_number_of_children()
        #
        # if self.mating_pool_size < self.crossover_operator.get_number_of_children():
        #     self.mating_pool_size = self.crossover_operator.get_number_of_children()

    def create_initial_solutions(self) -> List[S]:
        # jmetal 代码
        return [self.population_generator.new(self.problem)
                for _ in range(self.population_size)]

        # TZB 2022.1.21 为初始种群多样性，随机产生编码后微调
        # solution_list = []
        # number_of_same_first = int(self.population_size/self.problem.number_of_variables) # 编码首位item重复个数
        # item_index = 0  # 编码首位item索引值
        # for pop_i in range(self.population_size):
        #     solution = self.population_generator.new(self.problem)
        #     if pop_i < number_of_same_first * self.problem.number_of_variables:
        #         if item_index * number_of_same_first <= pop_i <= (item_index + 1) * number_of_same_first - 1 \
        #                 and solution.variables[0] != item_index :   # 交换编码两个位置的值，编码值为item_index的位 and 第0位
        #             target_index = solution.variables.index(item_index)
        #             temp_value = solution.variables[0]
        #             solution.variables[0] = solution.variables[target_index]
        #             solution.variables[target_index] = temp_value
        #         if pop_i == (item_index + 1) * number_of_same_first - 1:
        #             item_index += 1
        #     solution_list.append(solution)

        # return solution_list

    def evaluate(self, population: List[S]):
        return self.population_evaluator.evaluate(population, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def selection(self, population: List[S]):
        mating_pool = []

        # DE/rand/1 种群中随机选择3个体进化
        # for individual_index in range(self.population_size):
        #     self.selection_operator.set_index_to_exclude(individual_index)
        #     selected_solutions = self.selection_operator.execute(self.solutions)
        #     mating_pool = mating_pool + selected_solutions

        # DE/best/1
        front_size = 0
        if 'dominance_ranking' not in self.solutions[0].attributes: # 如果是初始解，未做非支配排序
            front_size = self.population_size
        else:
            # 获取前沿size，从front中选一个作为基准个体
            for individual_index in range(self.population_size):
                if self.solutions[individual_index].attributes['dominance_ranking'] == 0:
                    front_size += 1
                else:
                    break

        self.selection_operator.set_front_size(front_size)
        for individual_index in range(self.population_size):
            selected_solutions = self.selection_operator.execute(self.solutions)
            mating_pool = mating_pool + selected_solutions

        return mating_pool

    def reproduction(self, mating_population: List[S]) -> List[S]:
        # 一般交叉 number_of_parents_to_combine =2
        # 差分交叉 number_of_parents_to_combine =3

        # 差分变异操作，涉及个体数量
        number_of_parents_to_combine = self.mutation_operator.get_number_of_parents()
        # 整除关系：种群大小，单次操作个体数量，
        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception('Wrong number of parents')

        offspring_population = []

        for individual_index in range(self.population_size):
            parents = []
            # 按种群中的个体顺序，每number_of_parents_to_combine个一组，做变异操作
            for parents_individual_index in range(number_of_parents_to_combine):
                parents.append(mating_population[individual_index + parents_individual_index])
            trial = self.mutation_operator.execute(parents)  # 变异

            cross_inputs = [copy.deepcopy(self.solutions[individual_index]),
                            copy.deepcopy(trial[0])]
            cross_outputs = self.crossover_operator.execute(cross_inputs) # 交叉

            """
            TZB 2022.1.30 评价cross_outputs中2个解，并比较，太耗时了，先踢掉
            """

            # for cross_output in cross_outputs:
            #     self.problem.evaluate(cross_output)

            # dominance_flag = self.dominance_comparator.compare(cross_outputs[0],cross_outputs[1])

            # for solution in offspring:
            # if dominance_flag == 1:
            #     offspring_population.append(cross_outputs[0])
            # elif dominance_flag == -1:
            #     offspring_population.append(cross_outputs[1])
            # else:
            #     fist_obj_flag = self.dominance_comparator.compare_first_obj(cross_outputs[0],cross_outputs[1])
            #     if fist_obj_flag == 1:
            #         best_index = 0
            #     else:
            #         best_index = random.randint(0,1)
            #     offspring_population.append(cross_outputs[best_index])

            best_index = random.randint(0, 1)
            offspring_population.append(cross_outputs[best_index])  # 保留父代的开头部分

            if len(offspring_population) >= self.offspring_population_size:
                break

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        population.extend(offspring_population)

        population.sort(key=lambda s: s.objectives[0])

        return population[:self.population_size]

    def get_result(self) -> R:
        return self.solutions[0]

    def get_name(self) -> str:
        return 'Differential evolution'