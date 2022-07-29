import numpy as np
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import os
from datetime import date

from pymoo.algorithms.soo.nonconvex.ga import GA
from optimization import BinarySubsetSelectionSemanticPrimitivesProblem
from pymoo.core.result import Result


class AlgorithmMinimizer:
    def __init__(self, algorithm: GA,
                 proplem: BinarySubsetSelectionSemanticPrimitivesProblem,
                 checkpoint_path: str, n_max_gen: int) -> None:
        """
        Initialize GA algorithm

        :param algorithm: GA, algorithm to minimize
        :param proplem: BinarySubsetSelectionSemanticPrimitivesProblem, problem to optimize
        :param checkpoint_path: str, path to saved checkpoint (numpy saved algorithm)
        :param n_max_gen: int, max number of iterations
        """
        self.checkpoint_path = checkpoint_path
        self.algorithm = algorithm
        self.problem = proplem
        self.n_max_gen = n_max_gen

        self.seed = 1

        self.result = None

    def run_minimization(self, save_dir: str = "") -> Result:
        """
        Optimize algorithm and save results with checkpoint
        For saving details see https://pymoo.org/interface/result.html

        :param save_dir: str, path to save dir
        :return: Result, GA trained output
        """
        termination = get_termination("n_gen", self.n_max_gen)

        # load and optimize from checkpoint if possible
        if self.checkpoint_path:
            checkpoint, = np.load(self.checkpoint_path, allow_pickle=True).flatten()
            checkpoint.has_terminated = False
            res = minimize(self.problem,
                           checkpoint,
                           verbose=True,
                           seed=self.seed,
                           termination=termination)

        # start optimizing from the scratch
        else:
            res = minimize(self.problem,
                           self.algorithm,
                           verbose=True,
                           seed=self.seed,
                           termination=termination)

        # save results
        self.result = res
        if save_dir:
            self.save_results(save_dir=save_dir)

        return res

    def save_results(self, save_dir: str) -> None:
        """
        Saving results in more usable format.
        see https://pymoo.org/interface/result.html
        :param save_dir: str, path to save dir
        :return: None
        """

        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X.numpy"), self.result.X)
        np.save(os.path.join(save_dir, "F.numpy"), self.result.F)

        final_pop_path = os.path.join(save_dir, "final_pop")
        os.makedirs(final_pop_path, exist_ok=True)

        pop = self.result.pop
        np.save(os.path.join(final_pop_path, "X.numpy"), pop.get("X"))
        np.save(os.path.join(final_pop_path, "F.numpy"), pop.get("F"))

        today = date.today()
        today = today.strftime("%d_%m_%Y")
        np.save(os.path.join(save_dir, f"checkpoint_{str(self.n_max_gen)}_{today}"), self.algorithm)