from transformers import BertTokenizer, BertModel
import numpy as np
from fasttext import load_model
import torch

from typing import List, Any, Tuple, Dict

class FastTextWordVectorizer:
    def __init__(self, model_path: str = 'cc.en.300.bin') -> None:
        """
        FastText vectorizer
        :param model_path: str, path to bin model file
        """
        self.encoder = load_model(model_path)
        self.dims = self.encoder.get_dimension()

    def vectorize(self, words: List[str]) -> np.array:
        """
        Vectorize list of words into matrix
        :param words: list of str, words to vectorize
        :return: np.array, matrix of shape (len(words), embedding dim)
        """
        embedded_tokens = list(map(self.encoder.get_word_vector, words))
        embedded_tokens = np.stack(embedded_tokens)
        return embedded_tokens

class Word2VecWordVectorizer:
    def __init__(self, embedding_path: str, word_lems_stems: Any, eps: float = 1e-5) -> None:
        """
        Word2Vec vectorizer

        :param embedding_path: str, path to txt model file
        :param word_lems_stems: text preprocressing class, that when called will prprocess word (see BaseStemsLemms in word_preprocessing_utils.py)
        :param eps: float, value to fill the vector in case if word not in vocabulary
        """

        self.embedding_index = dict(
            self.get_coefs(*o.split(" ")) for o in open(embedding_path, encoding='utf-8', errors='ignore')
        )
        self.dim = len(list(self.embedding_index.values())[0])

        self.word_lems_stems = word_lems_stems
        self.eps = eps

    def get_coefs(self, word: List[str], *arr: np.array) -> Tuple[List[str], np.ndarray]:
        """Load word and its vector from line"""
        return word, np.asarray(arr, dtype='float32')

    def get_zeros(self) -> np.array:
        """return zero np array"""
        return np.zeros(self.dim)

    def handle_out_voc(self, word: str) -> np.array:
        """
        Process out of vocabulary words with stemming/lemmatization
        If worr is not in vocabulary, will output self.eps filled array
        :param word: str, word to process
        :return: np.array, representation vector
        """
        for processor in self.word_lems_stems:
            processed_word = processor(word)
            if processed_word in self.embedding_index:
                return self.embedding_index[processed_word]
        print(f"Could not find word {word} in voc, using eps + zero approach")
        return self.get_zeros() + self.eps


    def vectorize(self, words: List[str]) -> np.array:
        """
        Vectorize words
        :param words: list of str, words to vectorize
        :return: np.array, respresentation matrix of shape (num words, embedding dim)
        """
        vectors = [self.embedding_index[x] if x in self.embedding_index else self.handle_out_voc(x) for x in words]
        return np.stack(vectors)



class BertWordVectorizer:
    def __init__(self,
                 pretrained_name: str = 'bert-base-uncased',
                 max_len: int = 10,
                 device: str = 'cpu'
                 ) -> None:

        '''
        Class for encoding texts with pretrained BERT model

        :param pretrained_name: str, path to checkpoints. Default: 'bert-base-uncased'
        :param max_len: int, max seq length to use. All sequences, that are longer than max_len, will cut to this length.
        Ones, that are <= max_len - will be padded with 0
        :param device: str: 'cuda' or 'cpu' - what device to use during training
        '''

        self.bert_model = BertModel.from_pretrained(pretrained_name,
                                                    output_hidden_states=False,
                                                    output_attentions=False).to(device)
        self.bert_model.eval()

        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        self.max_len = max_len
        self.device = device

    def vectorize(self, texts: List[str]) -> Dict[str, np.array]:
        """
        Vectorize texts
        :param texts: list of str, texts
        :return: dict of np.array:
            {
                "cls": np.array - representation of CLS token
                "avg": np.array - average over all tokens` representations
            }

            For an empty input will output: {"cls": np.array([]), "avg":np.array([])}
        """

        # loading into batches of 512 and processing

        if not len(texts):
            return {"cls": np.array([]), "avg":np.array([])}
        bs = 4
        if len(texts) < bs:
            batches = [texts]
        else:
            batches = np.array_split(texts, len(texts) // bs)

        output = {
            "cls": [],
            "avg": []
        }

        with torch.no_grad():
            for b in batches:
                encoded_batch = list(map(self._encode_text, b))
                encoded_batch = torch.stack(encoded_batch, 0).to(self.device)
                last_hidden_state = self.bert_model(encoded_batch).last_hidden_state
                output["cls"].append(
                    last_hidden_state[:,0,:].cpu().numpy()
                )

                for i, seq_rep in enumerate(last_hidden_state):
                    output["avg"].append(
                        seq_rep.mean(dim=0).numpy()
                    )


        output["cls"] = np.vstack(output["cls"]).reshape(len(texts), -1)
        output["avg"] = np.vstack(output["avg"]).reshape(len(texts), -1)

        return output

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode texts with tokenizer
        :param text: str, text to encode
        :return: torch.Tensor, long tensor of indexes to put through model
        """
        tokens = self.bert_tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) < self.max_len:
            tokens += [0]*(self.max_len - len(tokens))

        return torch.Tensor(tokens).long()
