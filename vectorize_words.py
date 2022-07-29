import os
from conf.vectorization_configs import CONFIGS
from vectorization_utils import *

from typing import List, Any

def read_wordlist(path: str) -> List[str]:
    """
    read line separated words from file
    :param path: str, path to file
    :return: list of words
    """
    words = open(path, "r").read().split("\n")
    words = [w.lower() for w in words if w]
    return words

def save_wordlist_representations(
        save_dir: str,
        word_list: List[str],
        vectorizer: Any,
        save_name: str) -> None:
    """
    save words representations in the numpy file

    :param save_dir: str, path to save directory
    :param word_list: list of str, words to vectorize
    :param vectorizer: vectorizer object, has to have
    :param save_name: str, filename to save with ".npy" ending
    :return: None
    """

    representations = vectorizer.vectorize(word_list)
    if isinstance(representations, dict):
        for k, arr in representations.items():
            np.save(os.path.join(save_dir, f"{k}_{save_name}"), arr)
    else:
        np.save(os.path.join(save_dir, save_name), representations)


def run_vectorization(args):
    os.makedirs(args.save_dir, exist_ok=True)

    for c in CONFIGS:
        vectorizer = c["vectorizer"](**c["params"])
        for d in os.listdir(args.lists_dir):
            word_list = read_wordlist(os.path.join(args.lists_dir, d))
            os.makedirs(os.path.join(args.lists_dir, d), exist_ok=True)
            save_wordlist_representations(
                save_dir=os.path.join(args.save_dir, d), word_list=word_list,
                vectorizer=vectorizer, save_name=c["save_name"]
            )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Word Lists vectorization')

    parser.add_argument('--lists_dir', type=str,
                        default="wordlists",
                        help='path to dir with word lists. Each list should be a file with words new line separated')
    parser.add_argument('--save_dir', type=str,
                        default="wordlists/embeddings/",
                        help='path to where save embeddings. In the dir the dir for every word list will be created with corresponding embedding files')
