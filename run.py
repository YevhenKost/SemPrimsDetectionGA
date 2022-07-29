from optimization import BinarySubsetSelectionOptimizationFunctions, BinarySubsetSelectionSemanticPrimitivesProblem
from page_rank import DictPageRank
from algorithm_utils import BinarySubsetSelectionGeneticAlgorithm
from minimization_utils import AlgorithmMinimizer
from graph_utils import load_graph_dict
from postprocessing_utils import PopulationDecoder, load_decoding_dict

from multiprocessing.pool import ThreadPool
from joblib import load
import json, os
import numpy as np

from typing import List

from pymoo.core.problem import starmap_parallelized_eval



def get_sp_gen_unique_ids(cands: List[List[int]]) -> np.array:
    """
    Get unique vertexes from generated lists of semantic primitives
    :param cands: list of lists of ints, generated semantic primitives lists
    :return: np.array, unordered array of unique vertexes
    """
    unique_vals = set(sum(cands, []))
    cand_ids = np.array(unique_vals)
    return cand_ids



def fit_ga(args):

    graph_dict = load_graph_dict(
        json_graph_path=os.path.join(args.load_dir, "graph.json")
    )
    n_vals = max(graph_dict.keys())

    optim_pagerank = load(
        os.path.join(args.load_dir, "pagerank.pickle")
    )

    sp_gen_lists = json.load(open(args.sp_gen_lists, "r"))
    sp_gen_unique_ids = get_sp_gen_unique_ids(sp_gen_lists)

    optim_functions = BinarySubsetSelectionOptimizationFunctions(
        wpagerank=optim_pagerank,
        graph_dict=graph_dict,
        n_vals=n_vals,
        card_mean=int(args.card_mean),
        val_prank_fill=args.val_prank_fill,
        sq_card_diff=args.card_diff**2
    )

    pool = ThreadPool(args.n_threads)

    problem_params = {
        "n_var": n_vals,
        "n_obj": optim_functions.n_obj,
        "n_constr": optim_functions.n_constr,
        "xl": None,
        "xu": None,
        "optim_functions": optim_functions,
        "runner": pool.starmap,
        "func_eval": starmap_parallelized_eval
    }
    proplem = BinarySubsetSelectionSemanticPrimitivesProblem(**problem_params)


    algorithm = BinarySubsetSelectionGeneticAlgorithm.get_algorithm(
        sp_gen_lists=sp_gen_lists,
        sp_gen_unique_ids=sp_gen_unique_ids,
        pop_size=args.pop_size,
        max_mutate=args.max_mutate,
        min_mutate=args.min_mutate
    )


    minimizer = AlgorithmMinimizer(
        algorithm=algorithm,
        proplem=proplem,
        checkpoint_path=args.chp_path,
        n_max_gen=args.n_max_gen
    )
    minimizer.run_minimization(save_dir=args.save_dir)


    decoding_dict = load_decoding_dict(
        enc_dict_path=os.path.join(args.load_dir, "encoding_dict.json")
    )

    final_populations = np.load(os.path.join(args.save_dir, "final_pop", "X.numpy.npy"))
    PopulationDecoder.decode_binary_populations(
        populations=final_populations,
        decoding_dict=decoding_dict,
        save_dir=args.save_dir
    )


    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(
            vars(args), f
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GA Semantic Primitives Optim., Binary Subset Selection')

    parser.add_argument('--load_dir', type=str, default="wordnet_graph_StanzaLemm_SSCDefsDrop/graph.json",
                        help='path to dir, which contains graph.json,encoding_dict.json and pagerank.pickle files')
    parser.add_argument('--chp_path', type=str, default="",
                        help='path to checkpoint to continue from')
    parser.add_argument('--cands_path', type=str,
                        default="wordnet_graph_StanzaLemm_SSCDefsDrop/wordnet_graph_StanzaLemm_SSCDefsDrop_1000_candidates_random2.json",
                        help='path to generated Sem.Prims. json file')
    parser.add_argument('--n_threads', type=int, default=5,
                        help="Num threads to use (multiprocessing)")
    parser.add_argument('--val_prank_fill', type=float, default=-1.0,
                        help="Value to return for pagerank obj. if there is a cycle in graph")
    parser.add_argument('--pop_size', type=int, default=100,
                        help="pop_size for Algotithm")
    parser.add_argument('--card_diff', type=int, default=50,
                        help="Cardinality max difference")
    parser.add_argument('--card_mean', type=int, default=2800,
                        help="Cardinality upper bound")
    parser.add_argument('--save_dir', type=str, default="GA_fitted",
                        help="dir to save results")


    args = parser.parse_args()
    fit_ga(args)