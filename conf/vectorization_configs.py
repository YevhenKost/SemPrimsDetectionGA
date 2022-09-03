from vectorization_utils import FastTextWordVectorizer, Word2VecWordVectorizer, BertWordVectorizer
from word_preprocessing_utils import EnglishStemsLemms, SpanishStemsLemms

CONFIGS = [
        {
            "vectorizer": FastTextWordVectorizer,
            "params":{
                "model_path": "fastText_models/cc.en.300.bin",
            },
            "save_name": "ft",
            "output_filenames":  ["ft.npy"]
        },
        {
            "vectorizer": Word2VecWordVectorizer,
            "params": {
                "embedding_path": "word2vec_models/glove.6B/glove.6B.300d.txt",
                "word_lems_stems": EnglishStemsLemms(),
                "eps": 1e-5
            },
            "save_name": "w2v_glove.6B.300d",
            "output_filenames":  ["w2v_glove.6B.300d.npy"]
        },
        {
            "vectorizer": Word2VecWordVectorizer,
            "params": {
                "embedding_path": "word2vec_models/glove.840B.300d.txt",
                "word_lems_stems": EnglishStemsLemms(),
                "eps": 1e-5
            },
            "save_name": "w2v_glove.840B.300d",
            "output_filenames":  ["w2v_glove.840B.300d.npy"]
        },
        {
            "vectorizer": Word2VecWordVectorizer,
            "params": {
                "embedding_path": "word2vec_models/glove.42B.300d.txt",
                "word_lems_stems": EnglishStemsLemms(),
                "eps": 1e-5
            },
            "save_name": "w2v_glove.42B.300d",
            "output_filenames":  ["w2v_glove.42B.300d.npy"]
        },
        {
            "vectorizer": BertWordVectorizer,
            "params": {
                "pretrained_name": "huggingface_models/bert-base-uncased",
                "max_len": 30,
                "device": 'cpu'
            },
            "save_name": "bert_bert-base-uncased",
            "output_filenames":  ["avg_bert_bert-base-uncased.npy", "cls_bert_bert-base-uncased.npy"]
        },

    ]
