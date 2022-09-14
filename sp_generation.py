from graph_utils import DirectedGraph
from typing import Dict, List, Union, Tuple
import random, json, os
from tqdm import tqdm
from multiprocessing import Pool
from graph_utils import load_graph_dict, get_num_vertices

class PrimitivesCandidatesGenerator:
    """
    Class for Permutation-based Generation of List of Semantic Primitives
    """
    def __init__(self, graph: Dict[int, List], num_vertices: int) -> None:
        """
        :param graph: dict, edges dict: {vertex: [destination_vertex, destination_vertex, ...]}
        :param num_vertices: int, number of vertices in graph
        """

        self.graph = graph
        self.num_v = num_vertices

    def _randomize_order(self, seq: Union[List, Tuple]) -> Union[List, Tuple]:
        """Outplace order randomization"""

        return sorted(seq, key=lambda k: random.random())

    def generate_list(self, i: int) -> List[int]:
        """
        Generating list of SP
        :param i: int, iteration number
        :return: list of ints, list of SP, containing vertexes
        """

        # empy SP set
        candidates = []

        # randomize order of adding vertexes
        randomized_q = self._randomize_order(
            seq=list(self.graph.keys())
        )

        # create graph
        current_graph = DirectedGraph(self.num_v)

        # adding vertex to graph
        for v in randomized_q:
            for dest in self.graph[v]:
                current_graph.add_edge(v, dest)

            # check if vertex can be considered SP candidate
            if current_graph.has_cycle():
                candidates.append(v)
                current_graph.delete_edges_from_vertex(v)
        return candidates

    def generate(self, N: int= 10) -> List[List[int]]:
        """
        Generate N SP sets
        :param N: intm number of SP sets to generate
        :return: list of lists of int, generated SP lists. Each list is a list of vertexes
        """

        all_candidates = []
        for i in tqdm(range(N)):
            all_candidates.append(self.generate_list(i))
        return all_candidates

def save_SP_lists(args):

    random.seed(args.seed)

    graph = load_graph_dict(os.path.join(args.load_dir, "graph.json"))
    num_vertices = get_num_vertices(os.path.join(args.load_dir, "encoding_dict.json"))

    sp_generator = PrimitivesCandidatesGenerator(
        graph=graph,
        num_vertices=num_vertices
    )

    with Pool(args.n_cores) as p:
        output = list(tqdm(p.imap(sp_generator.generate_list, range(args.N)), total=args.N))
    with open(
        os.path.join(args.load_dir, f"candidates_{str(args.N)}_random{str(args.seed)}.json"), "w"
    ) as f:
        json.dump(output, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SP prob calc')

    parser.add_argument('--load_dir', type=str, default="graph_dir/",
                        help='path to dir where the results and graph are stored')
    parser.add_argument('--N', type=int,
                        default=1000,
                        help='Number of experiments to run')
    parser.add_argument('--n_cores', type=int,
                        default=12,
                        help='Num cores to use'
                        )
    parser.add_argument('--seed', type=int,
                        default=2,
                        help='random seed to use'
                        )

    args = parser.parse_args()
    save_SP_lists(args)
