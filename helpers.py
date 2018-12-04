import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *


class Helpers():

    def __init__(self):
        self.wnl = nltk.WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        """
        negating words list taken from qdap package for R
        https://github.com/trinker/qdapDictionaries
        """
        #removed two of the words because lemmatizing can't and won't turn them into can and won
        self.negating_words_lemmatized = set(["ain", "aren", "couldn", "didn", "doesn", "don", "hasn", "isn", "mightn", "mustn",
                                              "neither", "never", "no", "nobody", "nor", "not", "shan", "shouldn", "wasn", "weren", "wouldn"])
        self.negating_words_stemmed = set(["ain", "aren", "couldn", "didn", "doesn", "don", "hasn", "isn", "mightn", "mustn",
                                              "neither", "never", "no", "nobodi", "nor", "not", "shan", "shouldn", "wasn", "weren", "wouldn"])
        self.negating_words = set(["ain't", "aren't", "can't", "couldn't", "didn't", "doesn't", "don't", "hasn't", "isn't", "mightn't",
                                   "mustn't", "neither", "never", "no", "nobody", "nor", "not", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't"])
        
        self.refuting_words = set(['fake','fraud', 'hoax', 'false', 'deny', 'denies', 'despite', 'nope', 'doubt', 'bogus', 'debunk', 'prank', 'retract'])
        self.refuting_words_stemmed = set(['fake','fraud', 'hoax', 'fals', 'deni', 'despit', 'nope', 'doubt', 'bogu', 'debunk', 'prank', 'retract'])
        
        self.stop_words = set(stopwords.words('english'))
        self.pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]

    """
    takes word and stems or lemmatizes+makes lowercase
    in: string
    out: string
    """

    def normalize_word(self, w):
        return self.wnl.lemmatize(w).lower()

    def stem_word(self,w):
        return self.stemmer.stem(w).lower()

    """
    lemmatized, lowercased tokens
    in: string
    out: string list
    """

    def get_tokenized_lemmas(self, s):
        lemmas = [self.normalize_word(t) for t in nltk.word_tokenize(s)]
        return [i for i in lemmas if len(i) > 1]

    """
    lowercased tokens - for use w/ word embeddings
    in: string
    out: string list
    """

    def get_tokens(self, s):
        return [t.lower() for t in nltk.word_tokenize(s)]

    """
    Cleans a string: trimming, removing non-alphanumeric
    in: string
    out: string
    """

    def clean(self, s, lower=True):
        if lower:
            return re.sub('[\W_]+', ' ', s, flags=re.UNICODE).lower()
        else:
            return re.sub('[\W_]+', ' ', s, flags=re.UNICODE)

    """
    Removes stopwords from a list of tokens
    in: string list
    out: string list
    """

    def remove_stopwords(self, l, remove_negation=True):
        return [w for w in l if (w not in self.stop_words or (w in self.negating_words_lemmatized and not remove_negation))]

    """
    Function for getting lemmatized tokens in one step
    in: string
    out: string list
    """

    def get_clean_tokens(self, body, remove_negation=True, stem = False):
        clean_body = self.clean(body)
        tokens = self.get_tokenized_lemmas(clean_body)
        clean_tokens = self.remove_stopwords(tokens, remove_negation)
        if stem:
            stems = [self.stem_word(w) for w in clean_tokens]
            clean_tokens = [i for i in stems if len(i) > 1]
        return clean_tokens

    """
    stolen from SO: 
    stackoverflow.com/questions/25534214

    in: string representing POS tag
    out: bool
    """

    def is_noun(self, tag):
        return tag in ['NN', 'NNS', 'NNP', 'NNPS']

    def is_verb(self, tag):
        return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    def is_adverb(self, tag):
        return tag in ['RB', 'RBR', 'RBS']

    def is_adjective(self, tag):
        return tag in ['JJ', 'JJR', 'JJS']
