# 编码方式0/1/2.../m 根据包的数量

import os
from jmetal.problem.multiobjective.MOQMKP import MOQMKP_PER
from jmetal.algorithm.multiobjective import MODE
from jmetal.operator.crossover import PMXCrossover_TZB
from jmetal.operator.mutation import PDDEMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator.selection import DifferentialEvolutionSelection_best_1,DifferentialEvolutionSelection


# coding: permutation 隐含启发式（逐一装满每个背包）
problem = MOQMKP_PER(filename=os.path.abspath('../MOQMKP_data/TZB_20_3_100_1.txt'))

Pop_size = 100

algorithm = MODE(
    problem=problem,
    population_size = Pop_size,
    offspring_population_size = Pop_size,
    mutation = PDDEMutation(CR = 0.8),
    selection = DifferentialEvolutionSelection_best_1(),
    crossover = PMXCrossover_TZB(probability = 1),
    termination_criterion = StoppingByEvaluations(max_evaluations = Pop_size*10)
)

algorithm.run()

print("Run time is: ", algorithm.total_computing_time)

from jmetal.util.solution import get_non_dominated_solutions, print_variables_to_file, print_function_values_to_file,\
    print_stamp_to_file

front = get_non_dominated_solutions(algorithm.get_result())

# save to files
# 保存permutation，int 变量，目标值，约束违反情况
print_stamp_to_file('VAR.'+algorithm.get_name()+'.'+problem.get_name(), mode='w', stamp='---final---')
print_variables_to_file(front, 'VAR.'+algorithm.get_name()+'.'+problem.get_name(), mode='a+')
# 保存目标值
print_stamp_to_file('FUN.'+algorithm.get_name()+'.'+problem.get_name(), mode='a+', stamp='---final---')
print_function_values_to_file(front, 'FUN.'+algorithm.get_name()+'.'+problem.get_name(), mode='a+')

# 最终pf散点图
# import jmetal.lab.visualization
# plot_front = jmetal.lab.visualization.Plot(title='Pareto front approximation', axis_labels=['x', 'y', 'z'])
# plot_front.plot(front, label='MODE-MOQMKP', filename='MODE-MOQMKP', format='png')