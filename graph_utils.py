from collections import defaultdict
from typing import List, Dict, Union, Tuple
import json

class DirectedGraph:
    def __init__(self, num_vertices: int) -> None:
        """
        :param num_vertices: int, total number of vertices in the graph
        """
        self.graph = defaultdict(list)
        self.V = num_vertices

    def add_edge(self, u: int, v: int) -> None:
        """
        Adding edge from vertex u to vertex v
        :param u, v: int, vertices
        :return: None
        """
        self.graph[u].append(v)

    def delete_edges_from_vertex(self, v: int) -> None:
        """
        Deleting all edges from vertex, but keeping it as a destination
        :param v: int, vertex
        :return:
        """
        self.graph[v] = []

    def forget_vertex(self, v: int) -> None:
        """
        Removing vertex from graph: deleting outpoing and incoming edges with the given vertex
        :param v: int, vertex
        :return: None
        """
        self.delete_edges_from_vertex(v)
        for k,dests in self.graph.items():
            if v in dests:
                dests.remove(v)

    def isCyclicUtil(self, v: int, visited: List[bool], recStack: List[bool]) -> bool:
        """
        Util function for recursive check if graph has a cycle. Updates visited and recStack with given vertex and checks neighbours
        :param v: int, vertex
        :param visited: list of bools, track of which vertices were visited
        :param recStack: list of bools, track of path of edges
        :return: bool, True if cycle was detected, False otherwise
        """

        # Mark current node as visited and
        # adds to recursion stack
        visited[v] = True
        recStack[v] = True

        # Recur for all neighbours
        # if any neighbour is visited and in
        # recStack then graph is cyclic
        for neighbour in self.graph[v]:
            if visited[neighbour] == False:
                if self.isCyclicUtil(neighbour, visited, recStack) == True:
                    return True
            elif recStack[neighbour] == True:
                return True

        # The node needs to be poped from
        # recursion stack before function ends
        recStack[v] = False
        return False

    # Returns true if graph is cyclic else false
    def has_cycle(self) -> bool:
        """
        Check recursively if the the graph contains a cycle
        :return: bool, True if cycle was detected, False otherwise
        """
        visited = [False] * (self.V + 1)
        recStack = [False] * (self.V + 1)
        for node in range(self.V):
            if visited[node] == False:
                if self.isCyclicUtil(node, visited, recStack) == True:
                    return True
        return False


    def update_graph_dict(self, graph_dict: Dict[int, List], skip_vertices: Union[List, Tuple]=()) -> None:
        """
        Given dict of edges, build graph. Skip some vertices if necessary
        :param graph_dict: dict, {vertex: [destination_vertex, destination_vertex, ...]}, edges dict
        :param skip_vertices: list of ints: which vertex to ignore during graph construction
        :return:
        """
        for v, connections in graph_dict.items():
            if v not in skip_vertices:
                for dest in connections:
                    if dest not in skip_vertices:
                        self.add_edge(v, dest)

    def clear_graph(self) -> None:
        """
        Delete all vertexes and edges
        :return: None
        """
        self.graph = defaultdict(list)


def load_graph_dict(json_graph_path: str) -> Dict[int,List[int]]:
    """
    Load graph dict from json file. Json should have following format:
        {vertex_id (str): [edge_vertex (int), ...]}
    Make sure that the number of vertexes in file match the total number of vertexes
    :param json_graph_path: str, path to precalculated graph edges dict
    :return: dict, converted graph dict to {vertex_id (int): [edge_vertex (int), ...]}
    """
    graph_dict = json.load(
        open(json_graph_path, "r")
    )
    graph_dict = {int(k): v for k, v in graph_dict.items()}
    return graph_dict

def get_num_vertices(json_enc_dict_path: str) -> int:

    """
    Get maximum index from encoding dict
    :param json_enc_dict_path: str, path to encoding dict ({word: word_id})
    :return: int, num of vertices
    """

    enc_dict = json.load(open(json_enc_dict_path, "r"))
    return max(list(enc_dict.values()))
