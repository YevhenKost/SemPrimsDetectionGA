from vectorization_utils import FastTextWordVectorizer, Word2VecWordVectorizer, BertWordVectorizer
from word_preprocessing_utils import EnglishStemsLemms

CONFIGS = [
        {
            "vectorizer": FastTextWordVectorizer,
            "params":{
                "model_path": "/media/yevhen/HDD1/DataSets/fasstText_models/cc.en.300.bin",
                "pad_value": 0
            },
            "save_name": "ft",
            "output_filenames":  ["ft.npy"]
        },
        {
            "vectorizer": Word2VecWordVectorizer,
            "params": {
                "embedding_path": "/media/yevhen/HDD1/DataSets/word2vec/glove.6B/glove.6B.300d.txt",
                "word_lems_stems": EnglishStemsLemms(),
                "eps": 1e-5
            },
            "save_name": "w2v_glove.6B.300d",
            "output_filenames":  ["w2v_glove.6B.300d.npy"]
        },
        {
            "vectorizer": Word2VecWordVectorizer,
            "params": {
                "embedding_path": "/media/yevhen/HDD1/DataSets/word2vec/f47355dd5b267bd10f08671e513790690233c76a9ffd73aa915d78f894a8912e/glove.840B.300d.txt",
                "word_lems_stems": EnglishStemsLemms(),
                "eps": 1e-5
            },
            "save_name": "w2v_glove.840B.300d",
            "output_filenames":  ["w2v_glove.840B.300d.npy"]
        },
        {
            "vectorizer": Word2VecWordVectorizer,
            "params": {
                "embedding_path": "/media/yevhen/HDD1/DataSets/word2vec/glove.42B.300d.txt",
                "word_lems_stems": EnglishStemsLemms(),
                "eps": 1e-5
            },
            "save_name": "w2v_glove.42B.300d",
            "output_filenames":  ["w2v_glove.42B.300d.npy"]
        },
        {
            "vectorizer": BertWordVectorizer,
            "params": {
                "pretrained_name": "/media/yevhen/HDD1/huggingface_models/bert-base-uncased",
                "max_len": 30,
                "device": 'cpu'
            },
            "save_name": "bert_bert-base-uncased",
            "output_filenames":  ["avg_bert_bert-base-uncased.npy", "cls_bert_bert-base-uncased.npy"]
        },

    ]
