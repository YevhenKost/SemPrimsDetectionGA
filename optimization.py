import numpy as np
from pymoo.core.problem import ElementwiseProblem, looped_eval
from graph_utils import DirectedGraph

from page_rank import DictPageRank
from typing import Dict, List, Union

from multiprocessing.pool import ThreadPool


class BinarySubsetSelectionOptimizationFunctions:
    def __init__(self,
                 wpagerank: DictPageRank, graph_dict: Dict[int, List[int]],
                 val_prank_fill: float, sq_card_diff: float,
                 n_vals: int, card_mean: int) -> None:

        """
        Loading optimization and constraints functions for SP Binary Subset Selection Problems
        see https://pymoo.org/problems/definition.html

        :param wpagerank: DictPageRank, fitted DictPageRank model to score vertexes
        :param graph_dict: dict, graph dict of format {vertex_id (int): [edge_vertex (int), ...]}
        :param val_prank_fill: float, value to return for pagerank objective function if there is a cycle in graph
        :param sq_card_diff: float, possible value for set cardinality to differ. (set cardinality - n_max) ** 2 <= sq_card_diff **2
        :param n_vals: int, number of unique vertexes
        :param card_mean: int, maximum cardinality of output binary set
        """

        self.sq_card_diff = sq_card_diff
        self.val_prank_fill = val_prank_fill

        self.graph_dict = graph_dict
        self.n_vals = n_vals

        self.card_mean = card_mean
        self.wpagerank = wpagerank

        self.n_obj = len(self.get_metrics_list(np.zeros(n_vals)))
        self.n_constr = len(self.get_constraints_list(np.zeros(n_vals)))

        self.sq_card_diff = sq_card_diff


    def F_wpagerank_mean_exp(self, v: np.array, is_cycle: bool) -> float:
        """

        objective function

        f(binary subset) = np.exp(sum of pageranks of included vertexes) / Number of included vertexes if not is_cycle
                           self.val_prank_fill otherwise

        :param v: np.array of bool values, population to check
        :param is_cycle: bool, if there is a cycle in the graph after removing the vertexes that are included in v
        :return: float
        """

        if is_cycle:
            return self.val_prank_fill
        indexes = np.where(v)[0]
        if not len(indexes):
            return self.val_prank_fill
        ranks = [self.wpagerank[x] for x in indexes]
        return np.exp(sum(ranks)/len(indexes))

    def get_metrics_list(self, v: np.array, is_cycle:Union[bool, float] = True) -> List[float]:

        """
        List of metrics for optimization for ElementwiseProblem
        :param v: np.array of bool values, population to check
        :param is_cycle: bool, if there is a cycle in the graph after removing the vertexes that are included in v
        :return: list of floats, metrics
        """

        output = [
            -self.F_wpagerank_mean_exp(v, is_cycle),
        ]

        return output

    def F_set_cardinality_sq(self, v: np.array) -> float:
        """
        Calculate squared cardinality difference
        :param v: np.array of bool values, population to check
        :return: float
        """

        return -self.sq_card_diff + ((self.card_mean - np.sum(v)) ** 2)

    def get_constraints_list(self, v: np.array, is_cycle:Union[bool, float] = True) -> List[float]:
        """
        Contraints functions list
        :param v: np.array of bool values, population to check
        :param is_cycle: bool, if there is a cycle in the graph after removing the vertexes that are included in v
        :return: list of floats, metrics
        """

        return [
            self.F_set_cardinality_sq(v)
        ]

    def F_iscycle(self, v: np.array) -> float:
        """
        Check if there is a cycle in graph after removing vertexes, which are included in v
        :param v: np.array of bool values, population to check
        :return: float, 0 if there is no cycle, 1 if there is one
        """

        indexes = np.where(v)[0]
        if not len(indexes):
            return 1.0

        full_graph = DirectedGraph(num_vertices=self.n_vals)
        full_graph.update_graph_dict(graph_dict=self.graph_dict, skip_vertices=indexes)
        is_cycle = float(int(full_graph.has_cycle()))

        return is_cycle

    def get_constr_eq_list(self, v: np.array) -> List[float]:
        """
        Get contraints functions list that should equal 0
        :param v: np.array of bool values, population to check
        :return: list of floats, metrics
        """

        return [
            self.F_iscycle(v)
        ]

class BinarySubsetSelectionSemanticPrimitivesProblem(ElementwiseProblem):

    """Problem of Binary Subset Selection for Semantic Primitives Detection"""

    def __init__(self, n_var: int, n_obj: int, n_constr: int, xl: np.ndarray, xu: np.ndarray,
                 optim_functions: BinarySubsetSelectionOptimizationFunctions,
                 runner=Union[ThreadPool, None], func_eval: looped_eval = looped_eval):
        """
        See https://pymoo.org/problems/definition.html for parameters
        """

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu,
            runner=runner,
            func_eval=func_eval
        )
        self.optim_functions = optim_functions
        self.n_max = self.optim_functions.n_vals

    def _evaluate(self, x: np.array, out, *args, **kwargs):
        """see https://pymoo.org/interface/problem.html"""

        is_cycle = self.optim_functions.F_iscycle(x)

        out["F"] = self.optim_functions.get_metrics_list(x, is_cycle=is_cycle)
        out["G"] = self.optim_functions.get_constraints_list(x, is_cycle=is_cycle)




