from typing import TypeVar, List, Generator
from jmetal.algorithm.singleobjective.differential_evolution import DifferentialEvolution
from jmetal.config import store
from jmetal.core.problem import Problem
from jmetal.operator.selection import DifferentialEvolutionSelection,BinaryTournamentSelection
from jmetal.operator.selection import DifferentialEvolutionSelection_best_1
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.evaluator import Evaluator
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement, RemovalPolicyType
from jmetal.util.comparator import DominanceComparator, Comparator, MultiComparator
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.core.operator import Mutation, Crossover, Selection

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: MODE
.. moduleauthor:: TZB 2021.10.17
"""

class MODE(DifferentialEvolution[S, R]):

    def  __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator
                 ):
        super(MODE, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover = crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator
        )
        self.dominance_comparator = dominance_comparator
        self.front_size = 0

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        """ This method joins the current and offspring populations to produce the population of the next generation
            by applying the ranking and crowding distance selection.

            :param population: Parent population.
            :param offspring_population: Offspring population.
            :return: New population after ranking and crowding distance selection is applied.
            """
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        density_estimator = CrowdingDistance()

        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        solutions = r.replace(population, offspring_population)

        return solutions

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'MODE'
