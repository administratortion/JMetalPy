# 编码方式0/1/2.../m 根据包的数量

import os

from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.problem.multiobjective.MOQMKP import MOQMKP_PER
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.mutation import PermutationSwapMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator.selection import DifferentialEvolutionSelection

# coding: permutation 隐含启发式（逐一装满每个背包）
problem = MOQMKP_PER(filename=os.path.abspath('../MOQMKP_data/TZB_10_2_100_1.txt'))

algorithm = NSGAIII(
    problem=problem,
    population_size=10,
    reference_directions=UniformReferenceDirectionFactory(3, n_points=10),
    crossover = PMXCrossover(probability = 1.0),
    mutation = PermutationSwapMutation(probability = 1.0),
    termination_criterion=StoppingByEvaluations(max_evaluations=100)
)

algorithm.run()

print("Run time is: ", algorithm.total_computing_time)

from jmetal.util.solution import get_non_dominated_solutions, print_variables_to_file, print_function_values_to_file,\
    print_stamp_to_file

front = get_non_dominated_solutions(algorithm.get_result())

# save to files
# 保存permutation，int 变量，目标值，约束违反情况
print_stamp_to_file('VAR.'+algorithm.get_name()+'.'+problem.get_name(), mode='a+', stamp='---final---')
print_variables_to_file(front, 'VAR.'+algorithm.get_name()+'.'+problem.get_name(), mode='a+')
# 保存目标值
print_stamp_to_file('FUN.'+algorithm.get_name()+'.'+problem.get_name(), mode='a+', stamp='---final---')
print_function_values_to_file(front, 'FUN.'+algorithm.get_name()+'.'+problem.get_name(), mode='a+')

# 最终pf散点图
# import jmetal.lab.visualization
# plot_front = jmetal.lab.visualization.Plot(title='Pareto front approximation', axis_labels=['x', 'y', 'z'])
# plot_front.plot(front, label='MODE-MOQMKP', filename='MODE-MOQMKP', format='png')