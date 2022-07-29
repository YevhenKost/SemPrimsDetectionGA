from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

from typing import Union
import os, json
from tqdm import tqdm

from conf.vectorization_configs import CONFIGS

class PaddHungarianMethodEvaluation:

    """
    Wrapper of scipy Hungarian method with additional averaging and padding
    """

    minimize_metrics = [
        "cosine"
    ]

    pad_val = 0

    @classmethod
    def _padd_matrix(cls, M: np.array) -> np.array:
        """
        Padding matrix to a square matrix with cls.pad_val value
        :param M: np.array, matrix to padd of shape (a,b)
        :return: np.array of shape (max(a,b), max(a,b))
        """
        (a, b) = M.shape
        if a > b:
            padding = ((0, 0), (0, a - b))
        else:
            padding = ((0, b - a), (0, 0))
        return np.pad(M, padding, mode='constant', constant_values=cls.pad_val)


    @classmethod
    def calculate(cls, X: np.array, Y: np.array, metric: Union[str, None]="cosine") -> float:
        """
        Calculate evaluation metric.
        :param X: np.array, embeddings matrix of words of shape (M, D); M is a number of words, D is a dimensionality of embeddings
        :param Y:np.array, embeddings matrix of words of shape (V, D); V is a number of words, D is a dimensionality of embeddings
        :return: float, evaluation metric
        """

        # calc pair dists
        dist_matrix = cdist(
            X,
            Y,
            metric=metric
        )

        # normalize scipy implemented cosine metric
        if metric == "cosine":
            dist_matrix = 1 - dist_matrix

        # transpose if num rows < num cols (for correctness of hungarian algorithm)
        if dist_matrix.shape[0] < dist_matrix.shape[1]:
            dist_matrix = dist_matrix.reshape(dist_matrix.shape[1], dist_matrix.shape[0])

        # as hungarian method will minimize the metric, convert values to negatives if required
        if metric in cls.minimize_metrics:
            dist_matrix = -dist_matrix

        # padd and run through hungarian algorithm
        padded_dist_matrix = cls._padd_matrix(dist_matrix)
        row_ids_hung_algo, col_ids_hung_algo = linear_sum_assignment(
            cost_matrix=padded_dist_matrix,
            maximize=False
        )

        # calculate final result
        sum_loss = padded_dist_matrix[row_ids_hung_algo, col_ids_hung_algo].sum()

        # convert back from negatives if required
        if metric in cls.minimize_metrics:
            sum_loss = -sum_loss

        avg_loss = sum_loss/padded_dist_matrix.shape[0]
        return avg_loss


def compare_wordlists(args):

    metrics_dict = {}
    available_embs_filenames = sum([x["output_filenames"] for x in CONFIGS], [])

    for pred_wordlist_dir in os.listdir(args.pred_wordlist_embeddings_dir):
        for embedding_type in tqdm(available_embs_filenames):

            metrics_dict[pred_wordlist_dir] = {}
            metrics_dict[pred_wordlist_dir][embedding_type] = {}

            pred_wordlist_embs = np.load(
                os.path.join(args.pred_wordlist_embeddings_dir, pred_wordlist_dir, embedding_type)
            )

            for target_wordlist_filename in os.listdir(args.target_wordlist_dir):
                testlist_embs = np.load(
                    os.path.join(args.target_wordlist_dir, target_wordlist_filename, embedding_type)
                )

                score = PaddHungarianMethodEvaluation.calculate(
                    X=pred_wordlist_embs,
                    Y=testlist_embs,
                    metric=args.metric
                )

                metrics_dict[pred_wordlist_dir][embedding_type][target_wordlist_filename] = score

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, f"metrics_{args.metric}.json"), "w") as f:
        json.dump(metrics_dict, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculating metrics for different wordlists')

    parser.add_argument('--pred_wordlist_embeddings_dir',
                        type=str,
                        default="GA_trained/sp_embeddings/",
                        help='path to dir where the embeddings of the predicted word lists are stored')
    parser.add_argument('--target_wordlist_dir',
                        type=str,
                        default="wordlists/embeddings",
                        help='path to dir where the embeddings of the target word lists are stored')
    parser.add_argument('--save_dir', type=str,
                        default="GA_trained/sp_lists",
                        help='path to dir, to save the results in json file')
    parser.add_argument('--metric', type=str,
                        default="cosine",
                        help='metric of similarity to use')

    args = parser.parse_args()
    compare_wordlists(args)