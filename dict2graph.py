import json
from tqdm import tqdm

from stop_words import get_stop_words, StopWordError
import os

import stanza
from typing import Dict, Any, Set


class Dict2Graph:

    """
    Class for converting dictionary into graph
    """

    def __init__(self, stanza_lang: str, stop_words_lang:str, word_dictionary: Dict[str, Any], stanza_dir: str ="",
                 drop_self_cycles: bool=False, lemm_always: bool=True) -> None:
        """
        :param stanza_lang: str, lang to use for stanza
        :param stop_words_lang: str, lang to use for stop-words (see stop_words package)
        :param word_dictionary: dict of the following structure:
        {word: [
            {"definition": definition},
            {"definition": definition},...
                ]
                    }
        :param stanza_dir: str, path to dir, where stanza model are stored. Default: ""
        :param drop_self_cycles: bool, if to remove definitions from dict that contain the word they suppose to define
        :param lemm_always: bool: if True, lemmatize all the words in definitions. Otherwise lemmatization will applied
        only if the word is not in the vocabulary
        """

        self.ppl = stanza.Pipeline(
            lang=stanza_lang,
            dir=stanza_dir,
            processors='tokenize,lemma'
        )

        self.word_dictionary = word_dictionary
        self.drop_self_cycles = drop_self_cycles

        self.lemm_always = lemm_always
        self.stop_words_lang = stop_words_lang


    def get_filtered_set_tokens(self, definition: str) -> Set[str]:
        """
        Retrieve set of tokens from the str definition. Depending on lemm_always parameter, the lemmatization will be
        applied always or only if the word is not in vocabulary.
        Words that are not in vocabulary will be dropped.
        :param definition: str, definition to process
        :return: set of str, set of tokens (words)
        """

        doc = self.ppl(definition)

        if self.lemm_always:
            tokens = [word.lemma for sent in doc.sentences for word in sent.words]
            tokens = [t.lower() for t in tokens if t.lower() in self.word_dictionary]
            tokens = set(tokens)
        else:
            tokens = set()
            for sent in doc.sentences:
                for word in sent:
                    if word.text.lower() in self.word_dictionary:
                        tokens.add(word.text.lower())
                    elif word.lemma.lower() in self.word_dictionary:
                        tokens.add(word.lemma.lower())

        return tokens

    def get_encoding_dict(self) -> Dict[str, int]:
        """
        Building encoding dict for the given vocabulary of self.word_dictionary
        :return: {word: idx}
        """
        return {k.lower():v for v,k in enumerate(list(self.word_dictionary.keys()))}

    def get_from_word_edges(self, word: str) -> Set[str]:
        """
        Building edges from the given word. Self-Cycles will be dropped if self.drop_self_cycles was set to True
        :param word: str, word from vocabulary (self.word_dictionary)
        :return: set of ints, set of egdes (word, word in definitions)
        """
        all_edges = set()

        for def_dict in self.word_dictionary[word]:
            processed_def = self.get_filtered_set_tokens(
                    definition=def_dict["definition"]
                )

            if self.drop_self_cycles:
                if word not in processed_def:
                    all_edges = all_edges.union(processed_def)
            else:
                all_edges = all_edges.union(processed_def)

        return all_edges

    def build_graph(self, encoding_dict: Dict[str, int]=None) -> Dict[str, Any]:
        """
        Building a graph of dictionary
        :param encoding_dict: dict or None. Precalcuated encoding dict {word: id}. Of None, will be built based on the
        provided dict self.word_dictionary
        :return: dict with fields:
            encoding_dict: dict,  encoding dict that was either provided or built
            graph: dict, {word_id: [wrod_id, word_id, ...]}, graph edges
        """

        # load and drop stopwords
        try:
            sw = get_stop_words(self.stop_words_lang)
        except StopWordError:
            sw = []
        self.word_dictionary = {k:v for k,v in self.word_dictionary.items() if k.lower() not in sw}

        # build encoding dict if necessary
        if not encoding_dict:
            encoding_dict = self.get_encoding_dict()
        vertex_connections = {}

        # create edges
        for word in tqdm(self.word_dictionary):
            edges = list(self.get_from_word_edges(word=word))
            encoded_edges = [encoding_dict[x] for x in edges]
            vertex_connections[encoding_dict[word]] = encoded_edges

        return {"encoding_dict": encoding_dict, "graph": vertex_connections}


def build_dict(args):

    word_dictionary = json.load(args.word_dictionary_path)

    processor = Dict2Graph(
        stanza_dir=args.stanza_dir,
        stanza_lang=args.stanza_lang,
        stop_words_lang=args.stop_words_lang,
        word_dictionary=word_dictionary,
        drop_self_cycles=args.drop_self_cycles,
        lemm_always=args.lemm_always
    )

    output_dict = processor.build_graph()

    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, "encoding_dict.json"), "w") as f:
        json.dump(output_dict["encoding_dict"], f)
    with open(os.path.join(args.save_dir, "graph.json"), "w") as f:
        json.dump(output_dict["graph"], f)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Dict to Graph')
    parser.add_argument('--word_dictionary_path', type=str,
                        default="",
                        help='path to word dictionary in json')
    parser.add_argument('--stanza_dir', type=str,
                        default="",
                        help='path to dir, where stanza model are stored')
    parser.add_argument('--stanza_lang', type=str,
                        default="en",
                        help='lang to use for stanza')
    parser.add_argument('--stop_words_lang', type=str,
                        default="english",
                        help='lang to use for stop-words (see stop_words package)')
    parser.add_argument('--save_dir', type=str,
                        default="wordnet_StanzaLemm_NotAlwaysLemmSSC",
                        help='path, where to save results')
    parser.add_argument('--drop_self_cycles', type=bool,
                        default=True,
                        help='drop definitions containing word to define')
    parser.add_argument('--lemm_always', type=bool,
                        default=True,
                        help='Whether to always lemmatize words in definitions or only when it is not found in dict')

    args = parser.parse_args()
    build_dict(args)
