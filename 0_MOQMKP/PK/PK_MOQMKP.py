import os
import numpy as np
import pandas as pd
from typing import List
import jmetal.algorithm.multiobjective
from jmetal.algorithm.multiobjective import MODE, NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.core.quality_indicator import InvertedGenerationalDistance,HyperVolume
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.operator.selection import DifferentialEvolutionSelection,DifferentialEvolutionSelection_best_1
from jmetal.operator.mutation import PermutationSwapMutation, PDDEMutation
from jmetal.operator.crossover import PMXCrossover, PMXCrossover_TZB
from jmetal.problem.multiobjective.MOQMKP import MOQMKP_PER
from jmetal.util.termination_criterion import StoppingByEvaluations


def configure_experiment(problems: dict, n_run: int):
    jobs = []
    pop_size = 100
    max_evaluations = pop_size * 100

    for run in range(n_run):
        for problem_tag, problem in problems.items():
            problem = MOQMKP_PER(filename = os.path.abspath('../MOQMKP_data/TZB_20_3_100_1.txt'))

            # jobs.append(
            #     Job(
            #         algorithm = NSGAII(
            #             problem = problem,
            #             population_size = pop_size,
            #             offspring_population_size = pop_size,
            #             crossover = PMXCrossover(probability = 1.0),
            #             mutation = PermutationSwapMutation(probability = 1.0),
            #             termination_criterion = StoppingByEvaluations(max_evaluations = max_evaluations)
            #         ),
            #         algorithm_tag = 'NSGAII',
            #         problem_tag = problem_tag,
            #         getDatas = getDatas,
            #     )
            # )

            # jobs.append(
            #     Job(
            #         algorithm=NSGAIII(
            #             problem=problem,
            #             population_size=pop_size,
            #             reference_directions = UniformReferenceDirectionFactory(3, n_points = 90),
            #             crossover=PMXCrossover(probability=1.0),
            #             mutation=PermutationSwapMutation(probability=1.0),
            #             termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            #         ),
            #         algorithm_tag='NSGAIII',
            #         problem_tag=problem_tag,
            #         getDatas=getDatas,
            #     )
            # )

            jobs.append(
                Job(
                    algorithm=MODE(
                        problem=problem,
                        population_size = pop_size,
                        offspring_population_size = pop_size,
                        mutation = PDDEMutation(CR = 1.0),
                        crossover = PMXCrossover_TZB(probability = 1.0),
                        selection = DifferentialEvolutionSelection(),
                        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
                    ),
                    algorithm_tag = 'MODE_rand',
                    problem_tag=problem_tag,
                    run=run,
                )
            )

            jobs.append(
                Job(
                    algorithm = MODE(
                        problem = problem,
                        population_size = pop_size,
                        offspring_population_size = pop_size,
                        mutation = PDDEMutation(CR = 1.0),
                        crossover = PMXCrossover_TZB(probability = 1.0),
                        selection = DifferentialEvolutionSelection_best_1(),
                        termination_criterion = StoppingByEvaluations(max_evaluations = max_evaluations)
                    ),
                    algorithm_tag = 'MODE_best',
                    problem_tag = problem_tag,
                    run = run,
                )
            )

    return jobs

def read_solution(filepath: str = None, Algname: str = None) -> List:
    """
    Returns:
        object: solutions
    """
    solutions: dict={}
    with open(filepath) as file:
        index=0
        for line in file:
            vector=[float(x) for x in line.split()]
            solution={Algname+str(index): vector}
            solutions.update(solution)
            index+=1
    return solutions


def solve(sol_index, data_array):
    sol=data_array[:, sol_index]
    obj1_not_worse=np.where(sol[0]>=data_array[0, :])[0]
    obj2_not_worse=np.where(sol[1]>=data_array[1, :])[0]
    not_worse_candidates=set.intersection(set(obj1_not_worse), set(obj2_not_worse))

    obj1_better=np.where(sol[0]>data_array[0, :])[0]
    obj2_better=np.where(sol[1]>data_array[1, :])[0]
    better_candidates=set.intersection(set(obj1_better), set(obj2_better))

    dominating_solution=list(set.intersection(not_worse_candidates, better_candidates))
    if len(dominating_solution)==0:
        return True
    else:
        return False


def Get_PF_True():
    # 读入一个算例，所有算法，多次运行，解集，全部混合一块
    input_dir= 'data'
    solutions={}
    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            try:
                # Linux filesystem
                algorithm, problem=dirname.split('/')[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem=dirname.split('\\')[-2:]

            if 'FUN' in filename:
                solutions.update(read_solution(filepath=os.path.join(dirname, filename),\
                                               Algname=algorithm+filename.split('.')[1]))

    # 提取目标值部分
    df=pd.DataFrame(data=solutions).T
    data_labels=list(df.index)
    data_array=np.array(df).T

    # get PF_True
    dominating_set=[]
    for k in range(data_array.shape[1]):
        if solve(k, data_array):
            dominating_set.append(data_labels[k])

    # Save PF_True to file
    # 用于自动计算指标
    with open('../../resources/reference_front/MOQMKP.pf', 'w') as of:
        for i in range(len(dominating_set)):
            value=solutions.get(dominating_set[i])
            str_temp = ""  # pf文档只输出目标值
            # str_temp=dominating_set[i]+'\t'
            for j in range(len(value)):
                str_temp+=str(value[j])+'\t'
            of.write(str_temp)
            of.write('\n')
    # 自己查看用
    with open('MOQMKP.pf', 'w') as of:
        for i in range(len(dominating_set)):
            value=solutions.get(dominating_set[i])
            str_temp=dominating_set[i]+'\t'
            for j in range(len(value)):
                str_temp+=str(value[j])+'\t'
            of.write(str_temp)
            of.write('\n')


if __name__=='__main__':
    # Configure the experiments
    jobs = configure_experiment(problems={'MOQMKP': MOQMKP_PER}, n_run=11)  # jmetal方式

    # # Run the study
    output_directory = 'data'

    experiment = Experiment(output_dir = output_directory, jobs=jobs)
    experiment.run()

    Get_PF_True()

    # Generate summary file
    generate_summary_from_experiment(
        input_dir=output_directory,
        reference_fronts='..\\..\\resources\\reference_front',
        quality_indicators=[InvertedGenerationalDistance(),HyperVolume(reference_point = [1.0, 1.0, 1.0])]
        )
