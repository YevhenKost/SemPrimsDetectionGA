from evolution_utils import BinaryMutationFromProbs,BinarySamplingFromEmpirical, BinaryCrossover
from pymoo.algorithms.soo.nonconvex.ga import GA

import numpy as np
from typing import List

class BinarySubsetSelectionGeneticAlgorithm:
    @classmethod
    def get_algorithm(cls,
                      sp_gen_lists: List[List[int]],
                      sp_gen_unique_ids: np.array,
                      pop_size: int,
                      max_mutate: int,
                      min_mutate: int) -> GA:
        """
        Load algorithm to optimize
        :param sp_gen_lists: list of lists of ints, pregenerated lists of semantic primitives
        :param sp_gen_unique_ids: list of ints, unique vertexes that appeared in sp_gen_lists
        :param pop_size: int, population size (see https://pymoo.org/algorithms/soo/ga.html)
        :param max_mutate: int, max number of elements to mutate in population
        :param min_mutate: int, min number of elements to mutate in population
        :return: GA, initialized algorithm
        """
        sampling = BinarySamplingFromEmpirical(sp_gen_lists=sp_gen_lists)
        crossover = BinaryCrossover()
        mutation = BinaryMutationFromProbs(
            max_mutate=max_mutate,
            min_mutate=min_mutate,
            sp_gen_unique_ids=sp_gen_unique_ids
        )

        algorithm = GA(
            pop_size=pop_size,
            eliminate_duplicates=True,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )
        return algorithm