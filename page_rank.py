from sknetwork.ranking import PageRank
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any
from graph_utils import load_graph_dict
from joblib import dump
import json, os


class DictPageRank:
	def __init__(self, graph: Dict[int, List[int]], params: Dict[str, Any]) -> None:
		"""
		PageRank fitting and storing. The seed is fixed
		:param graph: dict, graph dict of format {vertex_id (int): [edge_vertex (int), ...]}
		:param params: dict, parameters for sknetwork.ranking.PageRank model
		"""

		self.graph = graph
		self.seeds = {0: 1}

		# check the vertexes
		# If there are missing vertexes in the graph, the adjustments should be made
		# vertexes will be reordered so the algorithm could be fit
		# On the inference step the decoding mechanism will be applied to output correct scores
		num_vert = max(list(self.graph.keys()))
		if list(range(num_vert)) == sorted(list(self.graph.keys())):
			self.row2value = None
		else:
			self.row2value = {i:k for i,k in enumerate(self.graph)}

		adjacency_matrix = self.build_adjacency_matrix(graph)

		self.scores = self.calc_pagerank(adjacency_matrix, params)

	def build_adjacency_matrix(self, graph: Dict[int, List[int]]) -> np.array:
		"""
		Build adjacency matrix from given graph
		:param graph: dict, graph dict of format {vertex_id (int): [edge_vertex (int), ...]}
		:return: np.array, adjacency matrix
		"""

		adjacency_matrix = np.zeros((len(graph), len(graph)), dtype='bool')

		for origin, dests in tqdm(graph.items()):
			origin = int(origin)
			for dest_vertex in dests:
				if self.row2value:
					adjacency_matrix[origin][dest_vertex] = 1
				else:
					adjacency_matrix[self.row2value[origin]][self.row2value[dest_vertex]] = 1

		return adjacency_matrix

	def calc_pagerank(self, adjacency_matrix: np.array, params: Dict[str, Any]) -> np.array:
		"""
		Fit PageRank algorithm on given adjacency matrix
		:param adjacency_matrix: np.array, square bool adjacency matrix
		:param params: dict, parameters for sknetwork.ranking.PageRank model
		:return: np.array of shape (num rows of adjacency_matrix), the page ranks scores per vertex
		"""

		return PageRank(**params).fit_transform(
			input_matrix=adjacency_matrix,
			seeds=self.seeds
		)

	def get_score(self, vertex: int) -> float:
		"""
		Calculating page rank score
		:param vertex: int, vertex to get a score for
		:return: float, the page rank score for a given vertex
		"""

		if self.row2value:
			return self.scores[int(self.row2value[vertex])]
		else:
			return self.scores[vertex]

	def __getitem__(self, item: int) -> float:
		"""
		Calculating page rank score
		:param vertex: int, vertex to get a score for
		:return: float, the page rank score for a given vertex
		"""

		return self.get_score(item)


def fit_prank(args):
	graph = load_graph_dict(os.path.join(args.load_dir, "graph.json"))
	params = json.load(open(args.fit_params_path, "r"))
	prank_model = DictPageRank(graph=graph, params=params)

	os.makedirs(args.save_dir, exist_ok=True)

	save_path_prank_model = os.path.join(args.load_dir, "pagerank.pickle")
	dump(prank_model, save_path_prank_model)

	save_path_prank_params = os.path.join(args.load_dir, "pagerank_params.json")
	with open(save_path_prank_params, "w") as f:
		json.dump(params, f)


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='PageRank fit and save')

	parser.add_argument('--load_dir',
						type=str,
						default="wordnet_graph_StanzaLemm_SSCDefsDrop/",
						help='path to dir where the graph.json is located'
						)
	parser.add_argument('--fit_params_path',
						type=str,
						default="conf/params_pagerank.json",
						help='path to json file where the parameters for PageRAnk fit is located')
	args = parser.parse_args()
	fit_prank(args)