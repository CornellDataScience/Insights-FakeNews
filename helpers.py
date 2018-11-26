import re
import nltk
from sklearn import feature_extraction


class Helpers():

    def __init__(self):
        self.wnl = nltk.WordNetLemmatizer()

    """
    takes word and lemmatizes/makes lowercase
    in: string
    out: string
    """


    def normalize_word(self, w):
        return self.wnl.lemmatize(w).lower()


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


    def remove_stopwords(self, l):
        return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


    """
    Function for getting lemmatized tokens in one step
    in: string
    out: string list
    """


    def get_clean_tokens(self, body):
        clean_body = self.clean(body)
        tokens = self.get_tokenized_lemmas(clean_body)
        clean_tokens = self.remove_stopwords(tokens)
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