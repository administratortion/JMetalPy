import logging
import os
import time
from pathlib import Path
from typing import List

from jmetal.core.solution import FloatSolution, Solution
from jmetal.util.archive import NonDominatedSolutionsArchive, Archive

LOGGER = logging.getLogger('jmetal')

"""
.. module:: solutions
   :platform: Unix, Windows
   :synopsis: Utils to print solutions.

.. moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


def get_non_dominated_solutions(solutions: List[Solution]) -> List[Solution]:
    archive: Archive = NonDominatedSolutionsArchive()

    for solution in solutions:
        archive.add(solution)

    return archive.solution_list


def read_solutions(filename: str) -> List[FloatSolution]:
    """ Reads a reference front from a file.

    :param filename: File path where the front is located.

    Returns:
        object:
    """
    front = []

    if Path(filename).is_file():
        with open(filename) as file:
            for line in file:
                vector = [float(x) for x in line.split()]

                solution = FloatSolution([], [], len(vector))
                solution.objectives = vector

                front.append(solution)
    else:
        LOGGER.warning('Reference front file was not found at {}'.format(filename))

    return front


def print_stamp_to_file(filename: str, stamp: str = '-', mode: str = 'w'):
    LOGGER.info('Output file (variables): '+filename)

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    with open(filename, mode) as of:
        for i in range(1):   # 重复次数
            of.write(stamp)
        of.write('\n')

def print_variables_to_file(solutions, filename: str, mode: str = 'w'):
    LOGGER.info('Output file (variables): '+filename)

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    if type(solutions) is not list:
        solutions = [solutions]

    with open(filename, mode) as of:  # w改写，a+追加
        for solution in solutions:
            for variables in solution.variables:
                of.write(str(variables)+' ')
            of.write('|'+" ")
            for variables_int in solution.variables_int:
                of.write(str(variables_int)+' ')
            of.write('|'+" ")
            for function_value in solution.objectives:
                of.write(str(function_value)+' ')
            of.write('|'+" ")
            for constraint_value in solution.constraints:
                of.write(str(constraint_value)+' ')
            of.write("\n")


def print_variables_to_screen(solutions):
    if type(solutions) is not list:
        solutions = [solutions]

    for solution in solutions:
        print(solution.variables[0])


def print_function_values_to_file(solutions, filename: str, mode: str = 'w'):
    LOGGER.info('Output file (function values): '+filename)

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    if type(solutions) is not list:
        solutions = [solutions]

    with open(filename, mode) as of:
        for solution in solutions:
            for function_value in solution.objectives:
                of.write(str(function_value)+'\t')
            of.write('\n')


def print_function_values_to_screen(solutions):
    if type(solutions) is not list:
        solutions = [solutions]

    for solution in solutions:
        print(str(solutions.index(solution))+": ", sep='  ', end='', flush=True)
        print(solution.objectives, sep='  ', end='', flush=True)
        print()
