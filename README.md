# Code to the Paper "Automatic Detection of Semantic Primitives Using Optimization Based on Genetic Algorithm"

# Setup 

1. Clone the repository: 
```commandline
git clone https://github.com/YevhenKost/SemPrimsDetectionGA.git
```

2. Install requirements
```commandline
pip install -r requirements.txt
```

3. Fill the configs
   1. Pagerank model fitting parameters ([conf/params_pagerank.json](conf/params_pagerank.json)). The parameters description can be found via the following link: [PageRank](https://scikit-network.readthedocs.io/en/latest/reference/ranking.html)
   2. Word Vectorization paths and save names ([conf/vectorization_configs.py](conf/vectorization_configs.py)). For each vectorizer provide required model paths on your local machine.

# Usage
1. Prepare the dictionary in a following format and save to json file. Making a specific directory for the dictionary to store all the results is suggested. For example: 
```python
import json, os

# load dictionary
my_dict = {
    "cat": [
        {"definition": "a very cute animal"},
        {"definition": "makes muuuuuur"}
    ],
    "buy": [
        {"definition": "exchange something for a money"}
    ]
}

# save to the dir
SAVE_DIR = "cat_but_directory/"
os.makedirs(SAVE_DIR, exist_ok=True)
with open(os.path.join(SAVE_DIR, "dictionary.json"), "w") as f:
   json.dump(my_dict, f)
```

2. Convert dictionary to directed graph. It can be achieved via the command (paths are taken from the previous example):
```commandline
python dict2graph.py --word_dictionary_path cat_but_directory/dictionary.json --stanza_dir LOADED_STANZA_MODELS/en --stanza_lang en --stop_words_lang english --save_dir cat_but_directory/ --drop_self_cycles true --lemm_always false
```

The arguments required:
   * --word_dictionary_path: path to the dictionary saved in json format (see previous example)
   * --stanza_dir: For lemmatization the stanza package is used. Can be "", than the stanza package will download everything based on the language, given in --stanza_lang. For model details, see [Pipeline](https://stanfordnlp.github.io/stanza/pipeline.html).
   * --stanza_lang: Language of dictionary. List of available languages can be found [here](https://stanfordnlp.github.io/stanza/available_models.html).
   * --stop_words_lang: Stop words language to use. List of available languages can be found [here](https://pypi.org/project/stop-words/).
   * --save_dir: path to a directory, where the grapg dict files will be stored: word encoding dictionary and graph edges dictionary in json formats. Suggested approach is to use the same directory as for the graph.
   * --drop_self_cycles: boolean, whether to delete the definitions, which have a word they suppose to define. For example, for the word "bark" the definition "to bark" will not be used during graph building.
   * --lemm_always: boolean, whether to use lemmatization only if the word is not in dictionary vocabulary or always.


For more details: 

```commandline
python dict2graph.py -h
```

3. Run Generation of Permutation-based Semantic Primitives Sets:
```commandline
python sp_generation.py --load_dir cat_but_directory/ --N 1000 --n_cores 12 --seed 2
```
Note, that it could take a while. For example, for a wordnet dictionary the generation of 1,000 SP lists took around a week with multiprocessing.
The command execution will save in the --load_dir a generated lists in the following format and filename:
```python

sp_sets_format = [
   [1,2,3], # sp set
   [10,2,5] # sp set
]

filename = f"candidates_{str(N)}_random{str(seed)}.json" # N and seed are taken from the arguments
```


The arguments required: 
   * --load_dir: path to directory, which contains <i>graph.json</i> file (generated on previous step). The generated SP lists will be saved here.
   * --N: int, number of SP lists to generate (there is no gurantee that they will be all unqiue).
   * --n_cores: int, how many cores to use during multiprocessing.
   * --seed: int, fix random seed.

<br>
For more details: 

```commandline
python sp_generation.py -h
```

4. Fit PageRank model
```commandline
python page_rank.py --load_dir cat_but_directory/ --fit_params_path conf/params_pagerank.json
```

The fitted model will be saved to <i>--graph_path</i>.

The arguments required:
   * --load_dir: path to directory, which contains <i>graph.json</i> file (generated on the first step). In this directory the trained pagerank model will be saved.
   * --fit_params_path: path to json file with pagerank parameters. See [conf/params_pagerank.json](conf/params_pagerank.json)

<br>
For more details: 

```commandline
python page_rank.py -h
```


5. Run algorithm
```commandline
python run.py --load_dir cat_but_directory/ --cands_path cat_but_directory/candidates_1000_random2.json --n_threads 8 --val_prank_fill -1.0 --pop_size 100 --card_diff 50 --card_upper 2800 --save_dir GA_fit_model
```

Algorithm results will be saved to save_dir. See [https://pymoo.org/interface/result.html](https://pymoo.org/interface/result.html).
The decoded results will be stored in save_dir/sp_wordlists/.

<br>
The arguments required:
   * --load_dir: path to directory, which contains <i>graph.json, encoding_dict.json</i> and <i>pagerank.pickle</i> files (generated on previous steps).
   * --cands_path: path to json file with generated lists of Sem.Prims. (see Section 3.)
   * --chp_path: path to .npy checkpoint (if you want to continue training). After the model training this checkpoint will be saved in the save_dir
   * --n_threads: int, number of cores to use for multiprocessing
   * --val_prank_fill: negative float, value to use to return for mean pagerank objective function if the cycle is still detected in the graph.
   * --pop_size: int, population size (see [here](https://pymoo.org/algorithms/soo/ga.html#nb-ga))
   * --card_diff: int, maximum possible cardinality deviation (constraint function: f(X) = (X - card_mean) ** 2 <= card_diff).
   * --card_mean: int, mean cardinality for the constraint (constraint function: f(X) = (X - card_mean) ** 2 <= card_diff).
   * --save_dir: path, where training args, checkpoint and results will be stored.

For more details see:

```commandline
python run.py -h
```

# Testing

1. Prepare word lists
Create a dir, where each word list should be in a text file with newline separated word

2. Fill up the preprocessing configs
Before that fill up the [conf/vectorization_configs.py](conf/vectorization_configs.py) and [word_preprocessing_utils.py](word_preprocessing_utils.py) files.
[word_preprocessing_utils.py](word_preprocessing_utils.py) supports preprocessing for English, Spanish and Ukrainian at the moment, but it is possible to add new classes for other langs.
In [conf/vectorization_configs.py](conf/vectorization_configs.py) fill up the stemming/lemmatization fields with the suitable classes.

3. Vectorize target word lists
```commandline
python vectorize_words.py --lists_dir wordlists/ --save_dir wordlists/embeddings/
```
The arguments required:
   * --lists_dir: path to directory, which contains word lists (see Section 1).
   * --save_dir: path, where the embeddings should be saved. Will generate a directory for each wordlist with the same name as file. In each dir the embeddings in .npy format will be saved.

4. Vectorize obtained word lists (see Section 5 of Usage)
```commandline
python vectorize_words.py --lists_dir GA_fit_model/sp_wordlists --save_dir GA_fit_model/sp_embeddings/
```

5. Calculate and save metrics
```commandline
python evaluate.py --pred_wordlist_embeddings_dir GA_fit_model/sp_embeddings --target_wordlist_dir wordlists/embeddings/ --save_dir GA_fit_model/ --metric cosine
```

The arguments required:
   * --pred_wordlist_embeddings_dir: path to directory, where the embeddings for generated populations are stored (see previous step).
   * --target_wordlist_dir: path to directory, where the embeddings for target word lists are stored (see step 3).
   * --metric: metric to use. See [https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) ``metric`` argument.
   * --save_dir: path, where the metrics should be saved. The json file will be generated: metrics_metric.json, where metric is the specified one.