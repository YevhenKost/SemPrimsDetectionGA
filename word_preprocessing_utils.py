from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

class BaseStemsLemms:
    def __init__(self):
        self.stemmers = []
        self.lemmatizers = []
        
    def get_stem_func(self):
        """
        Generator of stemm and lemm functions that can be applied on string
        """
        for s in self.stemmers:
            yield s.stem
        for l in self.lemmatizers:
            yield l.lemmatize

    def __iter__(self):
        return self.get_stem_func()

class EnglishStemsLemms(BaseStemsLemms):
    def __init__(self):
        """
        English Stemmers and Lemmatizers
        Will be generated one by one
        """
        super(EnglishStemsLemms, self).__init__()

        self.stemmers = [
            PorterStemmer(),
            LancasterStemmer(),
            SnowballStemmer("english")
        ]
        self.lemmatizers = [
            WordNetLemmatizer()
        ]