import copy
from abc import ABC, abstractmethod
from jmetal.core.problem import Problem
from jmetal.core.solution import PermutationSolution
from jmetal.util.comparator import Comparator
from jmetal.config import store
from jmetal.util.evaluator import Evaluator
from jmetal.util.observer import Observer


class NondominanceCompare:
	"""
		能够输出非支配比较结果方法的类
	"""
	def __init__(self, problem: Problem, dominance_comparator: Comparator = store.default_comparator, population_evaluator: Evaluator = store.default_evaluator):

		self.compare = dominance_comparator                         # 非支配比较关系类
		self.result = []                                            # -1为前支配后；0为互不支配；1后支配前
		self.target_associated_objectives = []						# 用于plotting使用
		self.problem = problem
		self.population_evaluator = population_evaluator			# problem内部的评估算法
		self.number_of_variables = self.problem.number_of_variables          # 染色体数目
		self.number_of_objectives = self.problem.number_of_objectives        # 目标值数目
		self.number_of_constraints = self.problem.number_of_constraints      # 约束数目

	def to_permutation_solution(self, target_associated_pair):
		"""

		Args:
			target_associated_pair:  相关联的一对数据集

		Returns:
			为这一对数据集分别格式化为solution，用来进一步得出比较结果
		"""
		solution = []

		for target_variables, associated_variables in target_associated_pair:

			tar_solution = PermutationSolution(
				number_of_variables=self.number_of_variables, number_of_constraints=self.number_of_constraints, number_of_objectives=self.number_of_objectives)
			tar_solution.variables = copy.deepcopy(target_variables)

			associated_solution = PermutationSolution(
				number_of_variables=self.number_of_variables, number_of_constraints=self.number_of_constraints, number_of_objectives=self.number_of_objectives)
			associated_solution.variables = copy.deepcopy(associated_variables)

			# 前者是单个目标解，后者是单个关联解，进行评估
			target_solution, associated_solution = \
				self.population_evaluator.evaluate([tar_solution, associated_solution], self.problem)

			solution.append([target_solution, associated_solution])
		return solution

	def get_compare_result(self, target_associated_pair):
		"""

		Args:
			target_associated_pair: 相关联的一对数据集

		Returns:
			输入比较结果，-1为前支配后；0为互不支配；1后支配前
			格式为： [target_solution, associate_solution, compare_result]
		"""
		final_result = []            # 每次都要初始化为空

		_target_associated_pair = copy.deepcopy(target_associated_pair)
		solution = self.to_permutation_solution(_target_associated_pair)

		for target, associated in solution:
			result = self.compare.compare(target, associated)
			final_result.append([target.variables, associated.variables, result])

		self.result = copy.deepcopy(final_result)
		return final_result

	def get_dataset_for_plotting(self, single_target_multi_associated):
		"""

		Args:
			single_target_multi_associated:

		Returns:
			输出适用于Plotting的数据集
		"""
		dataset = []

		solution1 = PermutationSolution(
			number_of_variables=self.number_of_variables,
			number_of_constraints=self.number_of_constraints,
			number_of_objectives=self.number_of_objectives)

		solution2 = solution1

		for target_variables, associated_variables in single_target_multi_associated:
			solution1.variables = target_variables
			self.population_evaluator.evaluate([solution1], self.problem)
			target_objective = solution1.objectives

			associated_objective = []
			for item in associated_variables:
				solution2.variables = item
				self.population_evaluator.evaluate([solution2], self.problem)
				associated_objective.append(solution2.objectives)

			dataset.append([target_objective, associated_objective])

		return dataset

