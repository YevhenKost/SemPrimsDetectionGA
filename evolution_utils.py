from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
import numpy as np

from typing import List

class BinarySamplingFromEmpirical(Sampling):
    def __init__(self, sp_gen_lists: List[List[int]]) -> None:
        """
        Sample populations from pregenerated ones
        :param sp_gen_lists: list of lists of ints, pre-generated semantic primitives
        """
        super(BinarySamplingFromEmpirical, self).__init__()
        self.sp_gen_lists = sp_gen_lists

    def _do(self, problem,
            n_samples, **kwargs):
        """
        see https://pymoo.org/operators/index.html and https://pymoo.org/operators/sampling.html for examples
        """
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        unused_ids = list(range(len(self.sp_gen_lists)))

        for k in range(n_samples):
            set_ind = np.random.choice(unused_ids)
            unused_ids.remove(set_ind)

            I = self.sp_gen_lists[set_ind]
            X[k, I] = True

        return X

class BinaryCrossover(Crossover):
    def __init__(self):
        """see https://pymoo.org/operators/index.html and https://pymoo.org/operators/crossover.html for the example"""
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        """see https://pymoo.org/operators/index.html and https://pymoo.org/operators/crossover.html for the example"""

        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            I = np.where((np.logical_xor(p1, p2)))[0]

            elements_keep = np.random.choice(["p1", "p2"], size=len(I))


            _X[0,k,I] = [p1[loc] if elements_keep[i] == "p1" else p2[loc] for i,loc in enumerate(I)]

        return _X

class BinaryMutationFromProbs(Mutation):
    def __init__(self, max_mutate: int, min_mutate: int, sp_gen_unique_ids: np.array):
        """

        Mutation based on empirical results

        see https://pymoo.org/operators/mutation.html

        :param max_mutate: int, max number of elements to mutate in population
        :param min_mutate: int, min number of elements to mutate in population
        :param sp_gen_unique_ids: np.array, unique vertexes that can mutate
        """
        super(BinaryMutationFromProbs, self).__init__()
        self.max_mutate = max_mutate
        self.min_mutate = min_mutate
        self.sp_gen_unique_ids = sp_gen_unique_ids


    def _do(self, problem, X, **kwargs):
        """see https://pymoo.org/operators/mutation.html"""
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_false = is_false[np.in1d(is_false, self.sp_gen_unique_ids)]

            num_false_change = np.random.randint(self.min_mutate, self.max_mutate)
            num_true_change = np.random.randint(self.min_mutate, self.max_mutate)

            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false, size=num_false_change)] = True
            X[i, np.random.choice(is_true, size=num_true_change)] = False

        return X