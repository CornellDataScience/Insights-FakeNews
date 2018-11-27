import nltk
from nltk import tokenize
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from scipy import spatial
import spacy
from helpers import Helpers
import queue

class FeatureEngineering(Helpers):

    def __init__(self):
        Helpers.__init__(self)
        "setup after spacy is installed: python -m spacy download en"
        self.nlp = spacy.load("en")
        self.analyzer = SentimentIntensityAnalyzer()

    """
    sentiment for single sentence
    in: string
    out: sentiment dict - SEE BELOW
    """

    def get_sentiment(self, sentence):
        return self.analyzer.polarity_scores(sentence)

    """
    average of sentiment scores for each sentence
    {"pos", "neg", "neu", "compound"}

    in: string w/ multiple sentences
    out: dictionary object with sentiment scores for each category
    """

    def sentiment_multi(self, paragraph):
        categories = ["pos", "neg", "neu", "compound"]
        sentence_list = tokenize.sent_tokenize(paragraph)
        paragraphSentiments = {"pos": 0, "neg": 0, "neu": 0, "compound": 0}
        for sentence in sentence_list:
            vs = self.get_sentiment(sentence)
            for cat in categories:
                paragraphSentiments[cat] += vs[cat]
        for cat in categories:
            paragraphSentiments[cat] /= len(sentence_list)
        return paragraphSentiments

    """
    union of strings in A and B
    in: 2 string lists
    out: string list
    """

    def shared_vocab(self, a, b):
        return list(set(a).union(set(b)))

    """
    cosine similarity of two bags of words

    in: 2 string lists - cannot be empty
    out: float
    """

    def bow_cos_similarity(self, a, b):
        vocab = self.shared_vocab(a, b)
        a_bow, b_bow = set(a), set(b)
        if len(a) == 0 or len(b) == 0:
            return -1
        a_vec = [(1 if i in a_bow else 0) for i in vocab]
        b_vec = [(1 if i in b_bow else 0) for i in vocab]
        return 1 - spatial.distance.cosine(a_vec, b_vec)

    """
    just a cosine similarity wrapper
    pls no zero vectors
    """

    def cosine_similarity(self, a, b):
        return 1 - spatial.distance.cosine(a, b)

    '''
    extracts the subject, verb, and object of a sentence, 
    including news headlines and body text
    in: string (sentence)
    out: triple of (subj, verb, obj)
    '''

    def extract_SVO(self, sent):
        subj = ""
        verb = ""
        obj = ""
        subjFound = False
        verbFound = False
        objFound = False

        tokenized_sent = self.nlp(sent)

        for token in tokenized_sent:
            if (token.dep_ == "nsubj" and subjFound == False):
                subj = token.text
                subjFound = True

            if (token.pos_ == "VERB" and verbFound == False):
                verb = token.text
                verbFound = True
            elif (token.head.pos_ == "VERB" and verbFound == False):
                verb = token.head.text
                verbFound = True

            if (token.dep_ == "dobj" or token.dep_ == "pobj" and objFound == False):
                obj = token.text
                objFound = True

        return (subj, verb, obj)

    '''root distance feature link to paper:
    http://aclweb.org/anthology/N/N16/N16-1138.pdf

    returns the average root distance among all three tokens in a trigram.
    If the trigram had a negative score of 0, simply return 1.0
    '''

    def find_avg_root_dist(self, sent):
        tokenized_sent = self.nlp(sent)
        num_toks = len(tokenized_sent)
        root = self.find_root(tokenized_sent)

        trigram_tok_lst = list(
            zip(tokenized_sent, tokenized_sent[1:], tokenized_sent[2:]))

        max_neg, trigram = self.find_most_neg_trigram(trigram_tok_lst)

        if (max_neg == 0):
            return (1.0, trigram)

        dist = 0.0

        for token in trigram:
            dist_to_tok = self.calc_root_dist(root, token, num_toks)
            dist = dist + dist_to_tok

        avg_dist = dist / len(trigram)

        return (avg_dist, trigram)

    '''
    calculates the root distance. In other words,
    this method performs a BFS from the root to the
    token node and returns the distance divided by the number 
    of tokens in order to keep all the values between 0 and 1
    '''

    def calc_root_dist(self, root, token, num_toks):
        if (root == None):
            return 1.0

        dist = 0.0
        q = queue.Queue(maxsize=0)
        visited = set()

        q.put(root)
        visited.add(root)

        while (not(q.empty())):
            curr_tok = q.get()

            dist = dist + 1.0

            for tok in curr_tok.children:
                if (not(tok in visited)):

                    if (tok == token):
                        return dist / num_toks

                    q.put(tok)
                    visited.add(tok)

        return dist / num_toks

    '''
    retrieves the root of a tokenized sentence
    '''

    def find_root(self, tokenized_sent):
        for tok in tokenized_sent:
            if (tok.dep_ == "ROOT"):
                return tok
        return None

    ''' 
    given a trigram list, 
    find the trigram that results in the most negativity
    returns the max_negative value, along with the trigram itself
    '''

    def find_most_neg_trigram(self, trigram_lst):
        max_neg = 0.0
        most_neg_trigram = trigram_lst[0]

        for trigram in trigram_lst:
            phrase = ""

            for tok in trigram:
                phrase = phrase + " " + tok.text

            polarity_scores = self.analyzer.polarity_scores(phrase)
            neg_val = polarity_scores["neg"]

            if (neg_val > max_neg):
                max_neg = neg_val
                most_neg_trigram = trigram

        return max_neg, most_neg_trigram

    """
    helper function for IDF's
    in: 2d list of strings
    out: dict of string -> idf score (float)
    """

    def build_idf_tokens(self, corpus, pos_tags=None):
        num_docs = len(corpus)
        vocabulary = Counter()
        for wordlist in list(corpus):
            if pos_tags is None:
                filtered_tokens = wordlist
            else:
                pos = nltk.pos_tag(wordlist)
                filtered_tokens = [x[0] for x in pos if (x[1] in pos_tags)]
            for i in set(filtered_tokens):
                vocabulary[i] += 1
        idf = {}
        for i in vocabulary:
            idf[i] = np.log(num_docs/(vocabulary[i]))
        return idf

    """
    get average tf-idf score of tokens in sentence
    if no idf, then use tf score
    must provide avg_idf if using idf!

    in: list of tokens, tf dictionary, idf dictionary, average of idf
    out: float
    """

    def score_sentence(self, sentence, tf, idf=None):
        acc = 0
        if idf == None:
            for token in sentence:
                acc += tf[token]
        else:
            avg_idf = idf["_avg"]
            for token in sentence:
                if token in idf:
                    acc += tf[token]*idf[token]
                else:
                    acc += tf[token]*avg_idf
        return acc/len(sentence)

    """
    extract metadata from sentence/body of text
    in: string
    out: dict
    """

    def process_sentence(self, s):
        # IGNORE LINT ERRORS, these methods will only be accessed in preprocessing
        clean_body = self.clean(s)
        tokens = self.get_tokenized_lemmas(clean_body)
        clean_tokens = self.remove_stopwords(tokens)
        svo = self.extract_SVO(" ".join(tokens))

        bigram = list(nltk.bigrams(clean_tokens))
        bigram_str = [x[0]+' '+x[1] for x in bigram]

        pos = nltk.pos_tag(clean_tokens)

        # list of words that belong to that part of speech
        nouns = [x[0] for x in pos if self.is_noun(x[1])]
        verbs = [x[0] for x in pos if self.is_verb(x[1])]
        adjectives = [x[0] for x in pos if self.is_adjective(x[1])]
        adverbs = [x[0] for x in pos if self.is_adverb(x[1])]

        vader_sentiment = self.get_sentiment(s)

        return {
            "tokens": clean_tokens,
            "bigrams": bigram_str,
            "nouns": nouns,
            "verbs": verbs,
            "adjectives": adjectives,
            "adverbs": adverbs,
            "sentiment": vader_sentiment,
            "svo": list(svo)
        }

    """
    return a dictionary of features
    see usage example:

    data - [headline:str, body_id:str/int]
    body_dict - dictionary of processed bodies
    idf - dictionary of idf scores

    NOTE: THIS IS VERY SLOW
    """

    def get_feats(self, data, body_dict, idf=None):
        headline, body_id = data[0], int(data[1])
        headline_data = self.process_sentence(headline)

        shared_common_nouns = len(set(headline_data['nouns']).intersection(
            set(body_dict[body_id]['common_nouns'])))
        shared_common_verbs = len(set(headline_data['verbs']).intersection(
            set(body_dict[body_id]['common_verbs'])))
        shared_common_tokens = len(set(headline_data['tokens']).intersection(
            set(body_dict[body_id]['common_tokens'])))
        shared_bigrams = len(set(headline_data['bigrams']).intersection(
            set(body_dict[body_id]['common_bigrams'])))

        shared_nouns_first = len(set(headline_data['nouns']).intersection(
            set(body_dict[body_id]['first_sentence']['nouns'])))
        shared_verbs_first = len(set(headline_data['verbs']).intersection(
            set(body_dict[body_id]['first_sentence']['verbs'])))
        shared_bigrams_first = len(set(headline_data['bigrams']).intersection(
            set(body_dict[body_id]['first_sentence']['bigrams'])))
        shared_tokens_first = len(set(headline_data['tokens']).intersection(
            set(body_dict[body_id]['first_sentence']['tokens'])))

        shared_nouns_sig = len(set(headline_data['nouns']).intersection(
            set(body_dict[body_id]['significant_sentence']['nouns'])))
        shared_verbs_sig = len(set(headline_data['verbs']).intersection(
            set(body_dict[body_id]['significant_sentence']['verbs'])))
        shared_bigrams_sig = len(set(headline_data['bigrams']).intersection(
            set(body_dict[body_id]['significant_sentence']['bigrams'])))
        shared_tokens_sig = len(set(headline_data['tokens']).intersection(
            set(body_dict[body_id]['significant_sentence']['tokens'])))

        # #adv and adj for stance
        # shared_adjectives_sig = len(set(headline_data['adjectives']).intersection(
        #     set(body_dict[body_id]['significant_sentence']['adjectives'])))
        # shared_adverbs_sig = len(set(headline_data['adverbs']).intersection(
        #     set(body_dict[body_id]['significant_sentence']['adverbs'])))
        # shared_adjectives_fst = len(set(headline_data['adjectives']).intersection(
        #     set(body_dict[body_id]['first_sentence']['adjectives'])))
        # shared_adverbs_fst = len(set(headline_data['adverbs']).intersection(
        #     set(body_dict[body_id]['first_sentence']['adverbs'])))

        #difference in sentiment
        sentiment_diff = {
            "pos": headline_data['sentiment']['pos']-body_dict[body_id]['sentiment']['pos'],
            "neg": headline_data['sentiment']['neg']-body_dict[body_id]['sentiment']['neg'],
            "neu": headline_data['sentiment']['neu']-body_dict[body_id]['sentiment']['neu'],
            "compound": headline_data['sentiment']['compound']-body_dict[body_id]['sentiment']['compound']
        }
        sentiment_diff_first = {
            "pos": headline_data['sentiment']['pos']-body_dict[body_id]['first_sentence']['sentiment']['pos'],
            "neg": headline_data['sentiment']['neg']-body_dict[body_id]['first_sentence']['sentiment']['neg'],
            "neu": headline_data['sentiment']['neu']-body_dict[body_id]['first_sentence']['sentiment']['neu'],
            "compound": headline_data['sentiment']['compound']-body_dict[body_id]['first_sentence']['sentiment']['compound']
        }
        sentiment_diff_sig = {
            "pos": headline_data['sentiment']['pos']-body_dict[body_id]['significant_sentence']['sentiment']['pos'],
            "neg": headline_data['sentiment']['neg']-body_dict[body_id]['significant_sentence']['sentiment']['neg'],
            "neu": headline_data['sentiment']['neu']-body_dict[body_id]['significant_sentence']['sentiment']['neu'],
            "compound": headline_data['sentiment']['compound']-body_dict[body_id]['significant_sentence']['sentiment']['compound']
        }

        headline_svo = headline_data['svo']
        body_fst_svo = body_dict[body_id]['first_sentence']['svo']
        body_sig_svo = body_dict[body_id]['significant_sentence']['svo']

        # cosine similarity - no verbs because relatively few per sentence
        cos_nouns_first = self.bow_cos_similarity(
            headline_data['nouns'], body_dict[body_id]['first_sentence']['nouns'])
        cos_bigrams_first = self.bow_cos_similarity(
            headline_data['bigrams'], body_dict[body_id]['first_sentence']['bigrams'])
        cos_tokens_first = self.bow_cos_similarity(
            headline_data['tokens'], body_dict[body_id]['first_sentence']['tokens'])

        cos_nouns_sig = self.bow_cos_similarity(
            headline_data['nouns'], body_dict[body_id]['significant_sentence']['nouns'])
        cos_bigrams_sig = self.bow_cos_similarity(
            headline_data['bigrams'], body_dict[body_id]['significant_sentence']['bigrams'])
        cos_tokens_sig = self.bow_cos_similarity(
            headline_data['tokens'], body_dict[body_id]['significant_sentence']['tokens'])

        return {
            'shared_nouns': shared_common_nouns,
            'shared_verbs': shared_common_verbs,
            'shared_bigrams': shared_bigrams,
            'shared_tokens': shared_common_tokens,

            'shared_nouns_fst': shared_nouns_first,
            'shared_verbs_fst': shared_verbs_first,
            'shared_bigrams_fst': shared_bigrams_first,
            'shared_tokens_fst': shared_tokens_first,

            'shared_nouns_sig': shared_nouns_sig,
            'shared_verbs_sig': shared_verbs_sig,
            'shared_bigrams_sig': shared_bigrams_sig,
            'shared_tokens_sig': shared_tokens_sig,

            'cos_nouns_sig': cos_nouns_sig,
            'cos_bigrams_sig': cos_bigrams_sig,
            'cos_tokens_sig': cos_tokens_sig,

            'cos_nouns_fst': cos_nouns_first,
            'cos_bigrams_fst': cos_bigrams_first,
            'cos_tokens_fst': cos_tokens_first,

            'svo_s_fst': int(headline_svo[0] == body_fst_svo[0]),
            'svo_v_fst': int(headline_svo[1] == body_fst_svo[1]),
            'svo_o_fst': int(headline_svo[2] == body_fst_svo[2]),

            'svo_s_sig': int(headline_svo[0] == body_sig_svo[0]),
            'svo_v_sig': int(headline_svo[1] == body_sig_svo[1]),
            'svo_o_sig': int(headline_svo[2] == body_sig_svo[2]),

            'sentiment_pos': sentiment_diff['pos'],
            'sentiment_neg': sentiment_diff['neg'],
            'sentiment_neu': sentiment_diff['neu'],
            'sentiment_compound': sentiment_diff_first['compound'],

            'sentiment_pos_fst': sentiment_diff_first['pos'],
            'sentiment_neg_fst': sentiment_diff_first['neg'],
            'sentiment_neu_fst': sentiment_diff_first['neu'],
            'sentiment_compound_fst': sentiment_diff_first['compound'],

            'sentiment_pos_sig': sentiment_diff_sig['pos'],
            'sentiment_neg_sig': sentiment_diff_sig['neg'],
            'sentiment_neu_sig': sentiment_diff_sig['neu'],
            'sentiment_compound_sig': sentiment_diff_sig['compound']
        }
