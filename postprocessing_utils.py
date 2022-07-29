import numpy as np
import json, os

from typing import Dict, List

class PopulationDecoder:
    """
    Decoder of word indexes into actual words
    """

    @classmethod
    def decode_binary_populations(cls, populations: np.array, decoding_dict: Dict[int, str], save_dir: str) -> None:
        """
        Decode binary vectors into words with provided decoding
        The word lists will be stored into save_dir. The new directory will be created inside save_dir: "sp_wordlists"
        In this directory the decoded word lists will be stored. Words will be line separated.

        :param populations: np.array of shape (D) or (N,D), where D is total number of words and N is number of elements in population
        Binary vectors of population. True on the position i means that word with index i should be included in the list
        :param decoding_dict: dict: {word_idx: word}, decoding dict
        :param save_dir: str, path to directory to save the decoded results
        :return: None
        """

        try:
            if len(populations.shape) == 1:
                populations = populations.reshape(1, -1)

            sp_wordlists_path = os.path.join(save_dir, "sp_wordlists")
            os.makedirs(sp_wordlists_path, exist_ok=True)

            for i, population in enumerate(populations):
                keep_indexes = np.where(population)[0]

                words = [decoding_dict[word_id] for word_id in keep_indexes]
                cls.write_words_to_file(
                    words=words,
                    filepath=os.path.join(sp_wordlists_path, f"sp_{str(i)}")
                )


        except Exception as e:
            print("Unable to perform decoding:")
            print(e)

    @classmethod
    def write_words_to_file(cls, words: List[str], filepath: str) -> None:
        """
        Creates file on filepath and writes to it words with "\n" separation

        :param words: list of str, words to store
        :param filepath: str, path to write the file
        :return: None
        """

        with open(filepath, "w") as f:
            for i, word in enumerate(words):
                f.write(word)
                if i < len(words) - 1:
                    f.write("\n")


def load_decoding_dict(enc_dict_path: str) -> Dict[int, str]:
    """
    Load decoding dict from encoding dict path

    :param enc_dict_path: path to encoding dict json of format: {word: word_idx}
    :return: decoding dict: {word_idx:word}
    """

    encoding_dict = json.load(
        open(enc_dict_path, "r")
    )

    decoding_dict = {v:k for k,v in encoding_dict.items()}

    return decoding_dict